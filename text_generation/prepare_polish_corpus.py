"""Original synthetic Polish dialogue fixture; not human-rated humor data."""
import argparse
from pathlib import Path

GIFTS = ('nowy sweter', 'ciepły pulower', 'wełniany szalik', 'wygodny szlafrok',
         'elegancki krawat', 'porządny płaszcz', 'miękki koc', 'ciepły golf')
OPENINGS = (
    'Kupię ci {gift}. Będzie ci ciepło i wygodnie.',
    'Zobacz, jaki ładny prezent: {gift}. Chcę ci go kupić.',
    'Mam pomysł na prezent dla ciebie: {gift}. Co ty na to?',
    'Przyda ci się {gift}. To prezent ode mnie, bez okazji.',
    'W sklepie czeka na ciebie {gift}. Pomyślałam o tobie.',
    'Chcę ci sprawić przyjemność. Może {gift}?',
    'Mam dla ciebie niespodziankę: {gift}. Podoba ci się?',
    'Nie marznij, kupię ci {gift}. Chcę o ciebie zadbać.',
)
ANSWERS = (
    'Kochanie, ja się potem nie wypłacę! Czy wdzięczność można rozłożyć na raty?',
    'Bardzo miło, ale czy do prezentu dostanę harmonogram spłaty wdzięczności?',
    'Zgoda, pod warunkiem że odsetki od wdzięczności nie będą rosły co miesiąc.',
    'Piękny prezent! Tylko powiedz, ile śniadań do łóżka wynosi pierwsza rata.',
    'Już się cieszę i trochę boję. Mój budżet na odwdzięczanie jest bardzo skromny.',
    'Dziękuję, kochanie! Czy przyjmujesz spłatę w przytuleniach i herbacie?',
    'Chętnie, ale potem będę ci winien tyle przysług, że muszę otworzyć arkusz kalkulacyjny.',
    'To bardzo miłe. Czy mogę spłacić ten dług do końca naszego wspólnego życia?',
    'Zaraz sprawdzę zdolność kredytową. W banku wdzięczności mam już spory debet.',
    'Kochanie, taki prezent to poważna inwestycja. Obiecuję wypłacać dywidendę w buziakach.',
    'Nie kuś mnie! Potem będę musiał gotować obiady aż do emerytury.',
    'Podoba mi się ten pomysł. Martwi mnie tylko drobny druk w umowie prezentowej.',
    'Biorę, jeśli cena obejmuje dożywotni serwis przytulania.',
    'Ależ ty o mnie dbasz! Ja się chyba odwdzięczę dopiero w następnym stuleciu.',
    'Dobrze, tylko ustalmy stałe oprocentowanie. Moje komplementy nie nadążają za inflacją.',
    'Brzmi cudownie. Czy w ramach spłaty mogę przez tydzień zmywać bez negocjacji?',
    'Przyjmuję z radością. Proszę tylko nie wysyłać mi faktury za bycie taką wspaniałą żoną.',
    'Jeśli to prezent, dlaczego mój wewnętrzny księgowy właśnie zemdlał?',
    'Och, kochanie! Za chwilę będę musiał zatrudnić doradcę do spraw małżeńskiej wdzięczności.',
    'Tylko nie mów, że promocja kończy się dziś. Ja potrzebuję miesiąca na zebranie komplementów.',
    'Zgadzam się, ale proszę o karencję. Pierwszą ratę wdzięczności wpłacę po porannej kawie.',
    'Dziękuję! Mogę oddać w naturze: kawa, kolacja i wynoszenie śmieci bez przypominania.',
    'To ja przygotuję kolację. Tak wielkiego długu nie można zostawić bez obsługi.',
    'Kochanie, serce mówi tak, a kalkulator prosi o chwilę na przemyślenie zobowiązań.',
)


def documents():
    return [f'Żona: {opening.format(gift=gift)} To drobiazg, naprawdę.\nMąż: {answer}'
            for gift in GIFTS for opening in OPENINGS for answer in ANSWERS]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path('corpora/polish-dialogues.txt'))
    a=p.parse_args()
    if a.out.exists():p.error('Output exists; choose a fresh --out path')
    docs=documents()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text('\n\n'.join(docs)+'\n',encoding='utf-8')
    print(f'Saved {len(docs)} synthetic dialogues to {a.out}')
    print(f'Characters per dialogue: {min(map(len,docs))}..{max(map(len,docs))}')
    print('Shared setups and punchlines cross document splits; not a novelty benchmark.')

if __name__=='__main__':main()
