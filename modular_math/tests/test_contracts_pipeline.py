import unittest
from contracts import (DerivativeMessage,arithmetic_answer,decode_message,message_ids,
                       gold_message,record,decode_product,parse_expression,encode)
from prepare_data import split_pairs
from pipeline import forward,score

class ContractTests(unittest.TestCase):
    def test_message_roundtrip_and_labels(self):
        for a,n in ((42,4),(-2,3),(1,1),(7,0),(0,0)):
            m=gold_message(a,n)
            self.assertEqual(decode_message(message_ids(m)),m)
            for role in ('structure','arithmetic'):
                row=record(role,a,n)
                self.assertEqual(len(row['input_ids']),len(row['labels']))
                self.assertEqual(row['labels'][len(row['prompt_ids'])-1:],row['answer_ids'])
            self.assertEqual(decode_product(arithmetic_answer(a*n)),a*n)
        self.assertEqual(parse_expression('-2*x**3'),(-2,3))
        self.assertEqual(parse_expression('x'),(1,1))
        self.assertEqual(parse_expression('7'),(7,0))
        with self.assertRaises(ValueError):parse_expression('__import__("os")')

    def test_split_exclusion(self):
        splits,_=split_pairs(199,5)
        seen=set()
        for values in splits.values():
            keys={tuple(sorted((abs(a),n))) for a,n in values}
            self.assertFalse(seen & keys);seen.update(keys)
        self.assertEqual(sum(map(len,splits.values())),2389)
        self.assertEqual(splits,split_pairs(199,5)[0])

class PipelineTests(unittest.TestCase):
    def test_actual_message_is_forwarded(self):
        # Deliberately wrong message: original input is 42*x**4, but module says 41*3.
        message=DerivativeMessage(41,3,2,False);received=[]
        def arithmetic(prompt):
            received.append(prompt);return arithmetic_answer(123)
        trace=forward(lambda prompt:message_ids(message),arithmetic,42,4)
        self.assertEqual(trace['arithmetic']['received_operands'],[41,3])
        self.assertEqual(trace['output']['coefficient'],123)
        result=score(trace,42,4)
        self.assertFalse(result['correct'])
        self.assertTrue(result['arithmetic_on_received_input_correct'])
        self.assertFalse(result['structure_contract_correct'])
        self.assertEqual(len(received),1)

    def test_correct_structure_wrong_arithmetic(self):
        trace=forward(lambda prompt:message_ids(gold_message(42,4)),
                      lambda prompt:arithmetic_answer(148),42,4)
        result=score(trace,42,4)
        self.assertTrue(result['structure_contract_correct'])
        self.assertTrue(result['structure_with_exact_arithmetic_correct'])
        self.assertFalse(result['arithmetic_on_received_input_correct'])
        self.assertFalse(result['correct'])

    def test_zero_bypass_and_malformed_output(self):
        def forbidden(prompt):raise AssertionError('Arithmetic must not run for predicted zero')
        trace=forward(lambda prompt:message_ids(gold_message(7,0)),forbidden,7,0)
        self.assertTrue(score(trace,7,0)['correct'])
        trace=forward(lambda prompt:message_ids(DerivativeMessage(42,4,3,True)),forbidden,42,4)
        self.assertFalse(score(trace,42,4)['correct'])
        trace=forward(lambda prompt:[],forbidden,42,4)
        self.assertEqual(trace['status'],'invalid_structure')
        trace=forward(lambda prompt:message_ids(gold_message(42,4)),lambda prompt:[],42,4)
        self.assertEqual(trace['status'],'invalid_arithmetic')

if __name__=='__main__':unittest.main()
