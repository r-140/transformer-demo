"""Pure orchestration. No reference labels are available to forward()."""
from contracts import (structure_prompt,arithmetic_prompt,decode_message,decode_product,
                       gold_message,normalized,expression,TOKENS)


def token_names(ids):
    return [TOKENS[i] if type(i) is int and 0<=i<len(TOKENS) else f'INVALID({i})' for i in ids]


def forward(structure_generate,arithmetic_generate,a,n):
    prompt=structure_prompt(a,n)
    ids=structure_generate(prompt)
    trace=dict(input=expression(a,n),structure=dict(prompt_tokens=token_names(prompt),
               generated_ids=ids,generated_tokens=token_names(ids)))
    try:message=decode_message(ids)
    except ValueError as e:
        trace.update(status='invalid_structure',error=str(e));return trace
    trace['structure']['message']=message.to_dict()
    if message.zero:
        trace.update(status='ok',arithmetic=dict(skipped=True,reason='Predicted zero flag'),
                     output=dict(coefficient=0,power=0,expression='0'))
        return trace
    # ONLY predicted message values cross this interface. No original a/n fallback.
    arithmetic_ids=arithmetic_prompt(message.coefficient,message.multiplier)
    result_ids=arithmetic_generate(arithmetic_ids)
    trace['arithmetic']=dict(skipped=False,prompt_tokens=token_names(arithmetic_ids),
        received_operands=[message.coefficient,message.multiplier],generated_ids=result_ids,
        generated_tokens=token_names(result_ids))
    try:product=decode_product(result_ids)
    except ValueError as e:
        trace.update(status='invalid_arithmetic',error=str(e));return trace
    coefficient,power=normalized(product,message.power)
    trace.update(status='ok',output=dict(coefficient=coefficient,power=power,
                                        expression=expression(coefficient,power)))
    return trace


def score(trace,a,n):
    """References are introduced AFTER inference, exclusively for diagnostics."""
    expected=gold_message(a,n)
    gold=normalized(a*n,max(0,n-1))
    result=dict(correct=False,structure_contract_correct=False,
                structure_with_exact_arithmetic_correct=False,
                arithmetic_called=False,arithmetic_on_received_input_correct=None,
                expected=dict(coefficient=gold[0],power=gold[1],expression=expression(*gold)))
    message=trace['structure'].get('message')
    if message is None:return result
    result['structure_contract_correct']=message==expected.to_dict()
    ideal=(0,0) if message['zero'] else normalized(message['coefficient']*message['multiplier'],message['power'])
    result['structure_with_exact_arithmetic_correct']=ideal==gold
    arithmetic=trace.get('arithmetic',{})
    result['arithmetic_called']=not arithmetic.get('skipped',True)
    if result['arithmetic_called']:
        result['arithmetic_on_received_input_correct']=(trace['status']=='ok' and
            trace['output']['coefficient']==message['coefficient']*message['multiplier'])
    if trace['status']=='ok':
        result['correct']=normalized(trace['output']['coefficient'],trace['output']['power'])==gold
    return result
