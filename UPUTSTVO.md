# 💎 MetaDrive Autonomous Driving Challenge --- Uputstvo 💎

## O izazovu

Vaš zadatak je da napravite **autonomnog agenta za vožnju** u MetaDrive
simulatoru. Agent treba da upravlja vozilom i donosi odluke u realnom
vremenu.

Cilj je balansirati četiri ključne stvari:

- **Bezbednost** --- bezsudara i izletanja sa puta
- **Glatkoća vožnje** --- bez naglih promena (jerk)
- **Napredak** --- što veća pređena distanca
- **Efikasnost** --- izbegavati nepotrebno gas/kočnica oscilovanje

Možete koristiti bilo koji pristup: od jednostavnih pravila do naprednih
ML modela.

------------------------------------------------------------------------

## Struktura projekta

`main.py` - Glavni program  
`game.py` - Glavna klasa koja upravlja okruženjem (env)  
`solution.py` - Ovde implementirate rešenje  
`control.py` - Tastatura kontroler  
`logger.py` - Logovanje i metrike  

Možete menjati sadržaj svih fajlova za potrebe istraživanja, debagovanja, testiranja, treninga... Ipak, evaluacija modela se vrši sa
originalnim sadržajem

**Fokus**:

- Biće evaluirano sa originalnim sadržajem `main.py`, `game.py`, `control.py` i `logger.py` skripti
- Možete dodavati nove fajlove, foldere...

------------------------------------------------------------------------

## Pokretanje

Instalacija: `pip install metadrive-simulator`

Pokretanje: `python main.py`

Kontrole za testiranje:

- **W** --- gas
- **S** --- kočnica
- **A / D** --- levo / desno
- **ESC / Q** --- izlaz

Preporuka: prvo probajte ručnu vožnju da razumete ponašanje auta.

------------------------------------------------------------------------

## Šta treba da implementirate

```python
class Solution:
    @property
    def config(self):
        return {
            "image_observation": False, "sensors": ["lidar"]
            }

    def do_iteration(self, simulator_output, user_input=None):
        return [steering, throttle]
```

### `config`

Ovde birate:

- koje senzore koristite (npr. lidar)
- dodatna podešavanja vozila (za senzore koje ste uključili)
- Ograničeni ste šta od konfiguracije možete da promenite (pogledajte funkciju `extract_config_from_solution()` u `main.py`)

### `do_iteration`

Poziva se na svakom koraku simulacije:

- ulaz: stanje iz simulatora, korisnički ulaz
- izlaz: `[steering, throttle]`

Ograničenja:

- Vrednosti moraju biti u opsegu **\[-1, 1\]**

------------------------------------------------------------------------

## ⚙️ Dozvoljeno

-   PyTorch, TensorFlow, NumPy...
-   Dodavanje novih fajlova

------------------------------------------------------------------------

## ❌ Zabranjeno

-   "Moj program radi samo sa mojom izmenjenom verzijom `main.py` skripte"
-   Eksterni API pozivi

------------------------------------------------------------------------

## Predaja rešenja

Šaljete:

- `solution.py`
- dodatne fajlove koje ste pravili (po potrebi)

------------------------------------------------------------------------

## Checklist

-   [ ] `config` implementiran
-   [ ] `do_iteration` implementiran
-   [ ] vraća validne akcije
-   [ ] nema pucanja aplikacije
-   [ ] testirano lokalno

------------------------------------------------------------------------

## Evaluacija

Vaše rešenje će biti ocenjeno kroz više simulacija sa različitim uslovima. Agent će biti testiran na različitim proceduralno generisanim mapama, sa varijacijama u okruženju i parametrima simulacije, kako bi se proverila opšta robusnost, a ne optimizacija za jedan scenario.

#### Kriterijumi

1. Bezbednost (najveći prioritet)

    - Broj sudara
    - Izletanja sa puta
    - Opasne situacije

    Teški penali za sudare

2. Glatkoća vožnje

    - Nagle promene gasa/kočenja
    - Nagle promene skretanja
    - Ukupan “jerk”

    Stabilna i kontrolisana vožnja dobija više poena

3. Napredak

    - Ukupna pređena distanca
    - Održavanje kretanja bez zastoja

    Agent koji stoji u mestu neće imati dobar skor

4. Efikasnost

    - Nepotrebno kočenje/gas
    - Oscilacije u kontroli

    Poželjna je “fluidna” vožnja bez stalnih korekcija

5. Robustnost

    - Performanse na različitim mapama
    - Ponašanje u neočekivanim situacijama
    - Stabilnost (bez crash-eva i bagova)

### Test slučajevi: degradirani senzori

Tokom evaluacije biće uključeni scenariji gde su senzorski podaci **delimično nepouzdani ili degradirani**, na primer:

- Lidar sa šumom (noise)
- Nedostajući podaci (prazni ili None segmenti)
- Kašnjenje u očitavanju
- Smanjena preciznost detekcije

Očekuje se da agent:

- prepozna nepouzdane podatke
- prilagodi ponašanje (npr. uspori)
- ili signalizira potrebu za preuzimanjem kontrole (driver takeover)



------------------------------------------------------------------------

## Bonus: ADAS (Advanced Driver Assistance Systems)

Za dodatne bodove možete implementirati ADAS funkcionalnosti koje unapređuju bezbednost i kvalitet vožnje.

### Moguće ADAS funkcije

1. Lane Keeping Assist (LKA)

    - Održava vozilo u centru trake
    - Koristi lidar ili poziciju vozila u odnosu na put
    - Blago koriguje steering bez naglih pokreta

2. Adaptive Cruise Control (ACC)

    - Automatski prilagođava brzinu
    - Smanjuje gas kada je prepreka ispred
    - Održava sigurnu udaljenost

3. Collision Avoidance

    - Detektuje prepreke unapred
    - Automatski koči ili skreće
    - Prioritet: bezbednost iznad svega

4. Smooth Steering Controller

    - Ograničava nagle promene steering-a
    - Smanjuje jerk i poboljšava stabilnost
    

    
    
Način na koji ste odlučili da ukombinujete korisnički ulaz i automatsko upravljanje ćete obrazložiti tokom prezentacije.

### Važna napomena (modularnost)

Iako u okviru zadatka već implementirate ove funkcionalnosti integrisano kroz autonomnu vožnju, dodatni poeni za ADAS se dodeljuju ako su ove komponente:

- modularno dizajnirane
- jasno razdvojene po funkciji
- mogu se nezavisno uključivati/isključivati

Primer:

- uključiti samo LKA
- isključiti Collision Avoidance
- testirati ACC nezavisno

Cilj je da sistem liči na **realne ADAS arhitekture**, gde svaka komponenta postoji kao **poseban “asistivni sloj”**, a ne kao jedna neodvojiva logika.

------------------------------------------------------------------------


## Bonus: Dashboard & HCI (Human–Computer Interaction)
Za dodatne bodove možete napraviti **dashboard (frontend UI)** koji simulira ekran u automobilu. Ovde nije fokus samo na prikazu podataka,
već na to **kako komunicirate sa vozačem**, tj. HCI (Human–Computer Interaction).
Cilj: vozač mora **brzo, jasno i bez zbunjivanja** da razume šta se dešava i da li treba da reaguje.


### Ključni HCI principi

1. **Jasnoća > količina informacija**  
Ne prikazujte sve - samo ono što je trenutno bitno.

2. **Hijerarhija važnosti**:  
Informacije treba da imaju prioritete:
    - Kritično - zahteva hitnu reakciju
    - Upozorenje - skrenuti pažnju
    - Informacija - stanje sistema

3. **Minimalno odvlačenje pažnje**  
Vozač ne sme da “čita UI” - treba da ga razume u deliću sekunde.

4. **Konzistentnost**  
Iste vrste upozorenja uvek izgledaju isto (boja, pozicija, stil).

### Saveti

- Ako vozač mora da razmišlja → UI nije dobar
- Jedna poruka u pravom trenutku > 5 istovremenih
- Fokus na reakciji, ne na estetici

------------------------------------------------------------------------

## Savet

Najbolji agent je stabilan, predvidiv i bez grešaka.

------------------------------------------------------------------------

Srećno 💎💎💎
