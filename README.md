# Analiza efektywności różnych architektur wieloagentowych w analizie dokumentów ubezpieczeniowych

_An effectiveness of various multi-agent architectures in the analysis of insurance documents_

**Praca Magisterska** — Polsko-Japońska Akademia Technik Komputerowych (PJATK), Wydział Informatyki

**\*Master's Thesis** — Polish-Japanese Academy of Information Technology (PJATK), Faculty of Computer Science\*

---

> **Uwaga dotycząca języka / Note on Language**
>
> Repozytorium zawiera dokumentację w dwóch językach. Wersja polska znajduje się na początku, a pełna wersja angielska poniżej.
>
> _This repository contains documentation in two languages. The Polish version is presented first, followed by the complete English version._

---

# WERSJA POLSKA (POLISH VERSION)

## Abstrakt

W pracy zbadano wpływ topologii systemów agentowych na jakość odpowiedzi na pytania wymagające analizy dokumentów ubezpieczeniowych — Ogólnych Warunków Ubezpieczenia (OWU) oraz dokumentów zawierających informacje o produkcie ubezpieczeniowym (IPID). Środowisko testowe wybrano ze względu na jego złożoność: specjalistyczny język prawniczo-ubezpieczeniowy łączy się z wielopoziomową strukturą warunków i wyłączeń rozproszonych w wielu paragrafach. Wymagało to od badanych architektur skutecznego wykorzystania mechanizmów wyszukiwania i wnioskowania.

Porównano sześć architektur opartych na dużych modelach językowych (_ang. Large Language Models_, LLM): deterministyczną, planer–wykonawca, router–specjalista, tablicową, hierarchiczną oraz ReAct. Bazę wiedzy systemu stanowiły 22 polskie dokumenty ubezpieczeniowe pochodzące od trzech głównych towarzystw ubezpieczeniowych. Opracowano autorski zestaw 90 pytań w czterech poziomach trudności: bardzo łatwym, łatwym, trudnym i bardzo trudnym — każde pytanie wymagało odniesienia się do treści zgromadzonych dokumentów. Łącznie przeprowadzono 540 uruchomień, które oceniono według wspólnego protokołu ewaluacyjnego uwzględniającego wierność, poprawność, zwięzłość, trafność pobieranych dokumentów, opóźnienie oraz zużycie tokenów.

Wyniki wskazują, że większa złożoność orkiestracji nie gwarantuje wyższej jakości odpowiedzi. Architektura router–specjalista osiąga najwyższą średnią wierność (0,811), a deterministyczna — najwyższą średnią poprawność (0,716), obie przy koszcie porównywalnym z najprostszymi układami. Architektury hierarchiczna i ReAct zużywają kilkukrotnie więcej tokenów bez proporcjonalnego przyrostu jakości — analiza jakościowa wskazuje, że dłuższe ścieżki przetwarzania prowadzą do utraty kontroli nad kontekstem i propagacji błędu między modułami. Sformułowano wnioski praktyczne dotyczące doboru architektury do przypadku użycia, opłacalności złożoności orkiestracji oraz roli przygotowania danych jako czynnika warunkującego skuteczność systemu niezależnie od topologii.

---

## Spis Treści

- [Pytania i Cele Badawcze](#pytania-i-cele-badawcze)
- [Architektury](#architektury)
- [Korpus Badawczy i Zbiór Danych](#korpus-badawczy-i-zbiór-danych)
- [Metodologia Eksperymentu](#metodologia-eksperymentu)
- [Metryki Ewaluacyjne](#metryki-ewaluacyjne)
- [Wyniki i Analiza Statystyczna](#wyniki-i-analiza-statystyczna)
- [Analiza Błędów](#analiza-błędów)
- [Rekomendacje dla Praktyków](#rekomendacje-dla-praktyków)
- [Stos Technologiczny](#stos-technologiczny)
- [Struktura Repozytorium](#struktura-repozytorium)
- [Współdzielone Narzędzia](#współdzielone-narzędzia)
- [Notatniki](#notatniki)
- [Instalacja i Uruchomienie](#instalacja-i-uruchomienie)
- [Rozbudowa Systemu](#rozbudowa-systemu)
- [Struktura Pracy Magisterskiej](#struktura-pracy-magisterskiej)

---

## Pytania i Cele Badawcze

### Pytania Badawcze (PB)

| ID      | Pytanie Badawcze                                                                                                                                                                                                       |
| :------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PB1** | Jak różne topologie agentowe wypadają pod względem wierności źródłom, poprawności odpowiedzi i trafności wyszukiwania dokumentów w kontrolowanym środowisku (ten sam model, narzędzia, korpus i protokół ewaluacyjny)? |
| **PB2** | Która topologia osiąga najkorzystniejszy stosunek jakości do kosztów, mierzony opóźnieniem (latency) i zużyciem tokenów na różnych poziomach trudności pytań?                                                          |
| **PB3** | W jaki sposób planowanie, rutowanie i dekompozycja zadań wpływają na wydajność w przypadku złożonych pytań wielowątkowych w porównaniu z jednoprzebiegowym (single-pass) wnioskowaniem?                                |
| **PB4** | Jakie wzorce błędów generuje każda z topologii i czy można je przypisać błędom wyszukiwania, błędom integracji kontekstu czy też iteracyjnemu dryfowi przepływu sterowania?                                            |

### Cele Badawcze (CB)

| ID      | Cel Badawczy                                                                                                                                                                     |
| :------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CB1** | Budowa powtarzalnej platformy orkiestracyjnej z wykorzystaniem LangGraph i Arize Phoenix dla sześciu architektur, traktując topologię jako jedyną zmienną niezależną.            |
| **CB2** | Stworzenie zbalansowanego zestawu testowego (90 pytań) opartego na treściach OWU/IPID, obejmującego trzech ubezpieczycieli, trzy kategorie produktów i cztery poziomy trudności. |
| **CB3** | Przeprowadzenie ewaluacji wszystkich architektur przy użyciu ujednoliconego protokołu obejmującego cztery metryki jakościowe i cztery metryki operacyjne.                        |
| **CB4** | Identyfikacja charakterystyki błędów poszczególnych topologii oraz sformułowanie praktycznych rekomendacji dla projektantów systemów produkcyjnych.                              |

---

## Architektury

Wszystkie sześć architektur zaimplementowano jako skompilowane grafy stanów w LangGraph. Operują one na tym samym obiekcie stanu `ExperimentState` (`TypedDict`), korzystają z tego samego rejestru narzędzi `TOOL_REGISTRY` i otrzymują ten sam obiekt konfiguracyjny `GraphConfig`. Topologia przepływu sterowania jest jedynym czynnikiem różnicującym.

| Architektura           | Mechanizm sterowania                                                                                                                 |
| :--------------------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| **Deterministyczna**   | Stały 8-węzłowy liniowy DAG; brak decyzji rozgałęziających.                                                                          |
| **Planer–Wykonawca**   | LLM generuje plan kroków (JSON); wykonawca realizuje je sekwencyjnie.                                                                |
| **Router–Specjalista** | LLM klasyfikuje intencję pytania do 1 z 8 kategorii; kieruje do dedykowanego podgrafu.                                               |
| **Tablicowa**          | Deterministyczny dyspozytor analizuje stan i kieruje do kolejnego węzła w oparciu o reguły (bez udziału LLM).                        |
| **Hierarchiczna**      | LLM dzieli pytanie na 1–3 podpytań; pętla robocza przetwarza każde z nich; syntezator łączy wyniki.                                  |
| **ReAct**              | Agent generuje Myśl + Akcję (Thought+Action) w każdym kroku; wykonawca uruchamia narzędzie, zwracając obserwację; maks. 15 iteracji. |

### Deterministyczna

Liniowa sekwencja 8 węzłów bez rozgałęzień: `parse_question → rewrite_query → retrieve_all → rerank → select_evidence → select_prompt → make_citations → generate_answer`. Zapewnia stabilność i najwyższą poprawność ogólną (0.716).

### Planer–Wykonawca

Model LLM na bazie pytania tworzy ustrukturyzowany plan działań w formacie JSON. Jeśli planowanie zawiedzie, system stosuje statyczny plan awaryjny odpowiadający architekturze deterministycznej.

### Router–Specjalista

Klasyfikator intencji (LLM) przypisuje pytanie do jednej z 8 kategorii tematycznych (np. _wyłączenia_, _limity_, _warunki_), kierując zapytanie do zoptymalizowanego podgrafu wyszukiwania. Osiąga najwyższą wierność (0.811) dla trudnych pytań.

### Tablicowa

Węzły komunikują się poprzez współdzielony stan. Regułowy dyspozytor (bypassing LLM) decyduje o kolejnym kroku. Wykazuje najniższy koszt operacyjny oraz zdolność do lokalnej autokorekty przy braku danych z pośrednich kroków.

### Hierarchiczna

Dekompozytor LLM dzieli złożone zapytanie na podpytania przetwarzane niezależnie przez wątki robocze. Głównym ograniczeniem jest bezwarunkowa dekompozycja prostych pytań, co drastycznie obniża wierność dla łatwych zadań.

### ReAct

Iteracyjna pętla Myśl-Akcja-Obserwacja. Pomimo braku przekroczenia sztywnego limitu 15 iteracji (maksymalnie zaobserwowano 11), architektura ta cierpi na dryf semantyczny w późniejszych krokach, uzyskując najniższą wierność (0.611).

---

## Korpus Badawczy i Zbiór Danych

### Dokumentacja Ubezpieczeniowa

Baza wiedzy zawiera 22 dokumenty (11 OWU oraz 11 IPID) od trzech ubezpieczycieli:

- **ERGO Hestia:** ERGO 7 Komunikacja, ERGO 7 Pozakomunikacyjne, ERGO 7 Podróż
- **PZU:** PZU Auto, PZU Dom, PZU Wojażer
- **Warta:** Autocasco (Komfort/Standard), Warta Dom (Komfort), Warta Travel

Dokumenty przetworzono na 5108 fragmentów (OWU: 4566, IPID: 542), wygenerowano embeddingi `text-embedding-3-large` (3072 wymiary) i zapisano w Qdrant.

### Zbiór Pytań Ewaluacyjnych

Plik `code/data/evaluation/questions/questions_20251130.csv` zawiera 90 ustrukturyzowanych pytań zbalansowanych trójwymiarowo: po 30 dla każdego ubezpieczyciela, po 30 dla kategorii produktowej oraz podział na cztery poziomy trudności: 9 pytań bardzo prostych i po 27 pytań na pozostałych trzech poziomach (Bardzo proste, Proste, Złożone, Bardzo złożone).

---

## Metodologia Eksperymentu

Eksperyment zaprojektowano jako kontrolowane badanie porównawcze, izolując topologię sterowania jako jedyną zmienną niezależną. Stałe środowiskowe:

- **Model językowy:** gpt-5 (dla wszystkich ról agentowych), temperatura 0.0.
- **Wyszukiwanie:** Pojedyncza kolekcja Qdrant, model osadzeń `text-embedding-3-large`.
- **Ewaluacja:** Framework Arize Phoenix w trybie LLM-as-judge (temperatura 0).

Eksperyment wykonano współbieżnie (32 wątki) w dniu 2026-03-13. Całkowity czas sekwencyjny wyniósł ok. 17.4 h, a rzeczywisty czas przetwarzania potoku (wall-clock) zamknął się w 1 h 47 min. Udane wykonanie odnotowano dla 539 z 540 przebiegów (jeden błąd typu 400 BadRequest w strukturze hierarchicznej).

---

## Metryki Ewaluacyjne

### Metryki Jakościowe (LLM-as-judge, skala [0.0, 1.0])

- **Wierność (Faithfulness):** Ocena stopnia zakotwiczenia wygenerowanej odpowiedzi w pobranym kontekście źródłowym.

- **Poprawność (Correctness):** Zgodność z ekspercką odpowiedzią wzorcową ze zbioru testowego.
- **Zwięzłość (Conciseness):** Brak redundantnych informacji i peryferyjnych dygresji.
- **Trafność Dokumentów (Document Relevance):** Semantyczne dopasowanie pobranych fragmentów tekstu do intencji pytania.

### Metryki Operacyjne

- Opóźnienie końcowe (Latency, ms), liczba tokenów wejściowych/wyjściowych, szacowany koszt finansowy (USD) oraz całkowita liczba wywołań narzędzi i iteracji grafu.

---

## Wyniki i Analiza Statystyczna

### Agregacja Metryk Jakościowych

| Architektura               | Wierność  | Poprawność | Zwięzłość | Trafność Dokumentów |
| :------------------------- | :-------- | :--------- | :-------- | :------------------ |
| **Router–Specjalista**     | **0.811** | 0.663      | 0.644     | 0.358               |
| **Tablicowa (Blackboard)** | 0.789     | 0.633      | 0.653     | 0.332               |
| **Planer–Wykonawca**       | 0.789     | 0.620      | 0.653     | 0.349               |
| **Deterministyczna**       | 0.778     | **0.716**  | 0.631     | **0.364**           |
| **Hierarchiczna**          | 0.697     | 0.588      | 0.676     | 0.344               |
| **ReAct**                  | 0.611     | 0.599      | **0.682** | 0.364               |

### Agregacja Metryk Operacyjnych

| Architektura           | Średnie opóźnienie (ms) | Tokeny wejściowe    | Tokeny wyjściowe  | Średni koszt (USD/run) |
| :--------------------- | :---------------------- | :------------------ | :---------------- | :--------------------- |
| **Tablicowa**          | 91,429 ± 19,646         | 3,004 ± 216         | 498 ± 145         | **0.00037**            |
| **Planer–Wykonawca**   | 84,255 ± 17,190         | 3,219 ± 261         | 580 ± 160         | 0.00040                |
| **Deterministyczna**   | 93,020 ± 17,599         | 2,988 ± 231         | 528 ± 160         | 0.00038                |
| **Router–Specjalista** | 98,913 ± 18,361         | 3,199 ± 247         | 519 ± 152         | 0.00039                |
| **Hierarchiczna**      | 147,096 ± 68,594        | 450,017 ± 1,080,756 | 115,761 ± 303,962 | 0.07109                |
| **ReAct**              | 181,936 ± 45,116        | 391,935 ± 419,373   | 58,548 ± 66,709   | 0.04320                |

---

## Analiza Błędów

W toku analizy jakościowej zidentyfikowano trzy profile błędów strukturalnych:

### Profil 1 — Odpowiedzi Niepełne (Deterministyczna, Tablicowa)

Merytorycznie poprawne względem pobranego kontekstu, lecz pomijające istotne wyłączenia odpowiedzialności ubezpieczyciela. Przyczyną jest _asymetria polaryzacji wektorowej_ – zapytania twierdzące klientów mapują się blisko klauzul ochrony, omijając semantycznie ujemne zapisy o wyłączeniach.

### Profil 2 — Odpowiedzi Nadmiarowe (Router–Specjalista, Planer–Wykonawca)

Potok wyszukiwania pobiera właściwe klauzule, lecz model syntezujący dokonuje tzw. _over-extension_, bezkrytycznie scalając detale z sąsiadujących, nieadekwatnych fragmentów (chunks) ze względu na bliskość w oknie kontekstowym.

### Profil 3 — Odpowiedzi Mieszane / Niespójne (Hierarchiczna, ReAct)

Błąd polegający na fuzji informacji z różnych produktów lub ubezpieczycieli. W strukturze hierarchicznej bezwarunkowy podział prostych pytań inicjuje niezależne procesy wyszukiwania w odległych rejonach kolekcji, a syntezator końcowy łączy je bez walidacji kontekstu produktowego. W architekturze ReAct, wielokrotne przepisywanie zapytań (query rewriting) prowadzi do dryfu semantycznego i pobierania danych z niespokrewnionych polis.

---

## Rekomendacje dla Praktyków

1. **Wdrożenie linii bazowej:** Jako architekturę pierwszego wyboru należy stosować podejście deterministyczne. Osiąga najwyższą poprawność ogólną (0.716), charakteryzuje się przewidywalnym profilem błędów (odpowiedzi niepełne zamiast halucynacji) i jest prosta w debugowaniu.
2. **Warunkowe wdrażanie routera:** Architektura router-specjalista powinna być aktywowana, gdy profil zapytań produkcyjnych zawiera wysoki odsetek pytań bardzo złożonych (zapewnia zysk +0.111 do wierności przy znikomym koszcie klasyfikacji).
3. **Izolacja typów dokumentów:** Należy bezwzględnie separować dane OWU i IPID na poziomie kolekcji wektorowej lub kategorycznych filtrów metadanych. Brak izolacji wywołuje konflikt priorytetów (algorytm wektorowy wyżej ocenia skrótowe, nieprecyzyjne tabele IPID niż wiążący tekst umowy OWU).
4. **Segmentacja zorientowana na strukturę prawa:** Granice fragmentów tekstu (chunking) powinny sztywno pokrywać się ze znacznikami redakcyjnymi aktów prawnych (ustępy, paragrafy) przy wielkości ok. 256 tokenów, co zapobiega zjawisku fuzji niespokrewnionych merytorycznie przepisów.
5. **Indeksowanie wyłączeń metodą roli:** Wprowadzenie flagi metadanych `role` (np. _ochrona_, _wyłączenie_, _definicja_) i wymuszenie pobierania min. jednego fragmentu o charakterze wyłączenia rozwiązuje problem asymetrii z Profilu 1.

---

## Stos Technologiczny

- **Orkiestracja i Grafy Stanów:** LangGraph v0.2.0
- **Integracja LLM:** LangChain & `langchain-openai` v1.2.10
- **Silnik Wnioskowania:** OpenAI API (`gpt-5`, `text-embedding-3-large`) v2.21.0
- **Ewaluacja i Obserwowalność:** Arize Phoenix v13 & OpenTelemetry auto-instrumentation
- **Bazy Danych:** Qdrant v1.15 (Wektorowa), PostgreSQL v14 (Repozytorium Phoenix)
- **Weryfikacja Struktur:** Pydantic v2.12.5
- **Przetwarzanie PDF:** `pypdfium2` & Pillow v5.4.0

---

## Struktura Repozytorium

```
master-thesis/
├── README.md                          # Dokumentacja główna
├── thesis/                            # Kody źródłowe pracy (LaTeX)
│   ├── thesis.tex                     # Główny plik kompilacji
│   ├── references.bib                 # Bibliografia BibTeX
│   └── thesis.pdf                     # Skompilowany dokument PDF
└── code/                              # Platforma eksperymentalna
    ├── sources/                       # Główny pakiet aplikacyjny
    │   ├── agents/                    # Implementacje grafów LangGraph
    │   │   ├── factory.py             # Fabryka grafów (build_graph)
    │   │   ├── state.py               # Definicja ExperimentState
    │   │   ├── deterministic.py       # Potok deterministyczny
    │   │   ├── router_specialist.py   # Potok router-specjalista
    │   │   └── ...                    # Pozostałe architektury
    │   ├── tools/                     # 8 asynchronicznych narzędzi (coroutines)
    │   ├── config/                    # Prompty, schematy Pydantic, AppConfig
    │   ├── runner.py                  # Współbieżny wykonawca eksperymentu
    │   └── evaluator.py               # Integracja z potokiem Phoenix
    ├── notebooks/                     # Notatniki Jupyter (NB_00 - NB_04)
    ├── data/                          # Dokumenty źródłowe, pytania i wyniki CSV
    ├── docker-compose.yml             # Konteneryzacja Qdrant, Phoenix i Postgres
    └── pyproject.toml                 # Definicja zależności uv
```

---

## Współdzielone Narzędzia

Każde z ośmiu narzędzi zaimplementowano jako asynchroniczną korutynę akceptującą kontrakt:
`async def tool_name(state: ExperimentState, *, config: GraphConfig, **kwargs) -> dict:`

- `question_parser`: Ekstrakcja metadanych zapytania (podmiot, produkt, kategoria intencji).
- `query_rewriter`: Generowanie wariantów zapytania (Direct, Step-back, HyDE).
- `retriever`: Wykonanie wyszukiwania w Qdrant z dynamicznym rozluźnianiem filtrów.
- `reranker`: Przeorganizowanie wyników (wyłączone w toku opisanego eksperymentu).
- `evidence_selector`: Deterministyczna selekcja top-k na podstawie dywersyfikacji.
- `citation_maker`: Konstrukcja jawnych obiektów referencyjnych (przypisów).
- `prompt_selector`: Mapowanie intencji pytania na 1 z 11 dedykowanych szablonów promptów.
- `answer_synthesizer`: Generowanie strukturyzowanej odpowiedzi finalnej w języku polskim.

---

## Notatniki

Wszystkie notatniki posiadają cechę idempotentności (pomijają kosztowne obliczenia, jeśli pliki wynikowe istnieją na dysku).

- `NB_00_visualize_agents.ipynb`: Generowanie diagramów Mermaid do formatów SVG/PDF (< 1 min).
- `NB_01_build_vectorstore.ipynb`: Ekstrakcja PDF, chunking i indeksowanie w Qdrant (2-3 h, OCR wizyjny).
- `NB_02_run_agents.ipynb`: Wykonanie 540 przebiegów testowych ( 2 h przy 32 wątkach).
- `NB_03_evaluate_agents.ipynb`: Uruchomienie potoku ewaluacji LLM-as-judge ( 1 h).
- `NB_04_visualize_results.ipynb`: Wygenerowanie wykresów statystycznych i zestawień (< 1 min).

---

## Instalacja i Uruchomienie

### Wymagania wstępne

- Python 3.13 lub nowszy

- Docker z Docker Compose
- Klucz API OpenAI z dostępem do modeli rodziny gpt-5

### Krok 1: Klonowanie i wejście do katalogu roboczego

```bash
git clone https://github.com/mchojna/master-thesis-pjatk.git
cd master-thesis-pjatk/code
```

### Krok 2: Uruchomienie kontenerów infrastruktury

```bash
docker compose up -d
```

_Wątek uruchomi Qdrant (port 6333), Arize Phoenix (port 6006) oraz bazę PostgreSQL (port 5432)._

### Krok 3: Synchronizacja środowiska wirtualnego

```bash
uv sync
```

### Krok 4: Konfiguracja zmiennych środowiskowych

Utwórz plik `.env` w katalogu `code/`:

```env
OPENAI_API_KEY=twój_klucz_api_openai
QDRANT_HOST=localhost
QDRANT_PORT=6333
PHOENIX_HOST=localhost
PHOENIX_PORT=6006
POSTGRES_URL=postgresql://phoenix:phoenix@localhost:5432/phoenix
```

### Krok 5: Egzekucja potoku

Uruchom serwer jupyter: `jupyter notebook` i wykonaj po kolei notatniki od `NB_01` do `NB_04`.

### Wywołanie programistyczne pojedynczego grafu

```python
import asyncio
from sources.agents.factory import build_graph
from sources.config.graph import GraphConfig


async def main():
    config = GraphConfig(pattern_name="deterministic")
    graph = build_graph(config)
    result = await graph.ainvoke({
        "question": "Ile wariantów ubezpieczenia oferuje ERGO Podróż?",
        "pattern_name": "deterministic",
    })
    print(result["answer_with_references"])


asyncio.run(main())
```

---

## Rozbudowa Systemu

### Dodanie nowej architektury

1. Utwórz plik `code/sources/agents/my_architecture.py` i zaimplementuj funkcję `build_graph(config: GraphConfig)` zwracającą obiekt `StateGraph`.
2. Zarejestruj moduł w słowniku `ALL_PATTERNS` w pliku `sources/agents/factory.py`.

### Rejestracja nowego narzędzia

1. Dodaj asynchroniczną funkcję w pakiecie `code/sources/tools/my_tool.py`.
2. Zadeklaruj referencję i opis w `sources/tools/__init__.py`:

   ```python
   from .my_tool import my_tool

   TOOL_REGISTRY["my_tool"] = my_tool
   TOOL_DESCRIPTIONS["my_tool"] = "Opis przekazywany do promptów planera/ReAct."
   ```

---

## Struktura Pracy Magisterskiej

Pełny tekst pracy w formacie PDF znajduje się w lokalizacji `thesis/thesis.pdf`. Do rekompilacji źródeł LaTeX wymagane jest środowisko TeX Live 2024+ oraz wykonanie poleceń: `cd thesis && latexmk -pdf thesis.tex`.

- **Rozdział 1: Wprowadzenie:** Uzasadnienie biznesowe, definicja pytań badawczych (PB1–PB4) oraz celów (CB1–CB4).
- **Rozdział 2: Podstawy teoretyczne:** Systemy agentowe, architektura RAG, specyfika języka dokumentów ubezpieczeniowych, przegląd literatury.
- **Rozdział 3: Metodologia:** Opis kontrolowanego eksperymentu, taksonomia zbioru danych, charakterystyka metryk.
- **Rozdział 4: Implementacja:** Szczegóły techniczne platformy, translacja teorii grafów na kod LangGraph, analiza narzędzi.
- **Rozdział 5: Eksperyment i wyniki:** Szczegółowa prezentacja danych empirycznych, testy statystyczne i porównania kosztowe.
- **Rozdział 6: Wnioski i podsumowanie:** Rozstrzygnięcie pytań badawczych, ograniczenia techniczne i kierunki dalszych badań.

---

_Kod źródłowy dystrybuowany jest na licencji MIT. Prawa autorskie do tekstu pracy magisterskiej © 2026 Michał Chojna. Wszelkie prawa zastrzeżone. Dokumenty OWU/IPID należą do ich wystawców i zostały użyte wyłącznie w celach naukowo-badawczych._

## <br><br>

---

<br><br>

# ENGLISH VERSION

## Abstract

This study examines the influence of agent orchestration topology on the quality of answers generated from Polish insurance documents. Six agent architectures — deterministic, planner–executor, router–specialist, blackboard, hierarchical, and ReAct — were implemented as LangGraph compiled state graphs and compared in a controlled experiment comprising 540 runs (90 evaluation questions across 6 architectures). All architectures shared an identical tool set, vector index, base language model, and evaluation protocol; topology was the sole independent variable.

The principal finding is that four architectures with fixed or shallowly-controlled execution (deterministic, planner–executor, router–specialist, blackboard) achieve faithfulness scores in the range 0.778–0.811 at a cost of 0.00037–0.00040 USD per query. The two iterative architectures (hierarchical, ReAct) incur costs 114–187 times higher while scoring lower on all four quality metrics. The results indicate that, in the single-hop insurance document domain, increased orchestration complexity does not produce proportional quality gains; targeted retrieval steering — as implemented by the router–specialist — provides a measurable, bounded advantage on complex questions at negligible additional cost.

---

## Table of Contents

- [Research Questions and Goals](#research-questions-and-goals-1)
- [Architectures](#architectures-1)
- [Dataset and Corpus Description](#dataset-and-corpus-description)
- [Experiment Design](#experiment-design-1)
- [Evaluation Metrics](#evaluation-metrics-1)
- [Results and Statistical Summary](#results-and-statistical-summary)
- [Failure Analysis](#failure-analysis-1)
- [Practitioner Recommendations](#practitioner-recommendations-1)
- [Technology Stack](#technology-stack-1)
- [Repository Structure](#repository-structure-1)
- [Shared Tools](#shared-tools-1)
- [Notebooks](#notebooks-1)
- [Setup and Execution](#setup-and-execution-1)
- [Extending the System](#extending-the-system-1)
- [Thesis Structure](#thesis-structure-1)

---

## Research Questions and Goals

### Research Questions (RQ)

| ID      | Question                                                                                                                                                                                       |
| :------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RQ1** | How do different agent topologies compare on source faithfulness, answer correctness, and document relevance in a controlled environment (same model, tools, corpus, and evaluation protocol)? |
| **RQ2** | Which topology achieves the most favourable quality–cost trade-off, measured by latency and token usage across question difficulty levels?                                                     |
| **RQ3** | How do planning, routing, and task decomposition affect performance on complex multi-clause questions compared with single-pass inference?                                                     |
| **RQ4** | What error patterns does each topology generate, and are they attributable to retrieval failures, context integration errors, or iterative control-flow drift?                                 |

### Research Goals (RG)

| ID      | Goal                                                                                                                                                     |
| :------ | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RG1** | Build a reproducible orchestration platform using LangGraph and Arize Phoenix with six architectures, treating topology as the sole independent variable |
| **RG2** | Create a balanced test set of 90 questions on OWU/IPID content, covering three insurers, three product categories, and four difficulty levels            |
| **RG3** | Evaluate all architectures using a unified protocol comprising four quality metrics and four operational metrics                                         |
| **RG4** | Identify the failure characteristics of each topology and derive practical recommendations for production system designers                               |

---

## Architectures

All six architectures are implemented as LangGraph compiled state graphs. They operate on the same `ExperimentState` TypedDict, invoke the same `TOOL_REGISTRY`, and receive the same `GraphConfig` at runtime. The control-flow topology is the only differentiating factor.

| Architecture          | Polish name        | Control mechanism                                                                                                   |
| :-------------------- | :----------------- | :------------------------------------------------------------------------------------------------------------------ |
| **Deterministic**     | Deterministyczna   | Fixed 8-node linear DAG; no branching decisions                                                                     |
| **Planner–Executor**  | Planer–wykonawca   | LLM generates a JSON step plan; executor follows it sequentially                                                    |
| **Router–Specialist** | Router–specjalista | LLM classifies question intent into one of eight categories; routes to one of four specialist sub-pipelines         |
| **Blackboard**        | Tablicowa          | Deterministic rule-based dispatcher inspects state and routes to the next eligible node (no LLM for routing)        |
| **Hierarchical**      | Hierarchiczna      | LLM decomposes the question into 1–3 sub-questions; worker loop processes each; synthesiser merges results          |
| **ReAct**             | ReAct              | Agent emits Thought + Action at each step; executor dispatches tool; observation is fed back; maximum 15 iterations |

### Deterministic pipeline

The deterministic architecture follows a fixed sequence of eight nodes without branching: `parse_question → rewrite_query → retrieve_all → rerank → select_evidence → select_prompt → make_citations → generate_answer`. It achieves the highest correctness score (0.716) and a stable execution trace.

### Planner–Executor

The planner LLM receives the question and tool descriptions, then produces an ordered JSON list of tool names. The executor invokes each tool in sequence. If planning fails, a static eight-step fallback plan equivalent to the deterministic pipeline is automatically used.

### Router–Specialist

The router LLM classifies the question into one of eight intent categories (faq, description, coverage, exclusions, conditions, limits, comparison, claims), which are mapped to one of four specialist retrieval sub-pipelines. This architecture achieves the highest faithfulness (0.811), particularly on very complex questions (0.815).

### Blackboard

A deterministic dispatcher inspects the current state at each cycle and routes to the next unfilled node according to a fixed rule table. The architecture requires no LLM for routing and achieves the lowest cost (0.00037 USD/query). It displays partial self-correction capabilities when an intermediate node produces empty results.

### Hierarchical

A decomposer LLM splits the question into 1–3 sub-questions. A worker loop processes each sub-question through a full retrieval and synthesis mini-pipeline. Unconditional decomposition of single-aspect questions degrades faithfulness on simple inputs (0.556 on very-easy questions).

### ReAct

The controller emits a Thought and selects an Action at each iteration; the executor dispatches the corresponding tool and loops back. The 15-iteration guard was not triggered (maximum observed count was 11). Repeated query scope drift in later iterations produces the lowest overall faithfulness score (0.611).

---

## Dataset and Corpus Description

### Corpus of Insurance Documents

The knowledge base comprises 22 Polish insurance documents — 11 OWU (Ogólne Warunki Ubezpieczenia) and 11 IPID (Informacyjny Dokument Produktu Ubezpieczeniowego) — from three insurers across three product categories:

- **ERGO Hestia:** ERGO 7 Komunikacja, ERGO 7 Pozakomunikacyjne, ERGO 7 Podróż
- **PZU:** PZU Auto, PZU Dom, PZU Wojażer
- **Warta:** Autocasco Komfort/Standard, Warta Dom/Komfort, Warta Travel

The resulting 5,108 text chunks (OWU: 4,566, IPID: 542) are embedded with `text-embedding-3-large` (3,072 dimensions) and stored in Qdrant with eight metadata payload indices.

### Evaluation Question Set

The question set is stored in `code/data/evaluation/questions/questions_20251130.csv`. It contains 90 questions balanced across three dimensions simultaneously: 30 per insurer, 30 per product category, and a distribution across 4 difficulty levels (9 very simple, 27 simple, 27 complex, 27 very complex).

---

## Experiment Design

The experiment was designed as a controlled comparative study with topology as the sole independent variable. The following factors were held constant across all runs:

- **Base language model:** gpt-5 (all roles), temperature 0.0.
- **Vector store:** Single Qdrant collection `insurance_documents`, `text-embedding-3-large`.
- **Evaluation protocol:** Arize Phoenix LLM-as-judge, identical prompt definitions across all topologies.

The experiment was executed on 2026-03-13 using a thread pool of 32 workers. Wall-clock time was 1 h 47 min; total sequential CPU/LLM processing time was approximately 17.4 h. 539 of 540 runs completed successfully (1 failure recorded in the hierarchical framework due to a 400 BadRequest error).

---

## Evaluation Metrics

### Quality Metrics (LLM-as-judge, Arize Phoenix)

All metrics are scored on a continuous scale from 0.0 to 1.0.

- **Faithfulness:** Whether every claim in the generated answer is grounded in the retrieved context.
- **Correctness:** Correspondence between the generated answer and the ground-truth reference answer.
- **Conciseness:** Degree to which the answer avoids unnecessary repetition or peripheral detail.
- **Document Relevance:** Relevance of cited document chunks to the question intent.

### Operational Metrics

- Latency (ms), input/output token counts, estimated financial cost (USD), and total counts of tool invocations and internal graph iterations.

---

## Results and Statistical Summary

### Quality Metrics by Architecture

| Architecture          | Faithfulness | Correctness | Conciseness | Document Relevance |
| :-------------------- | :----------- | :---------- | :---------- | :----------------- |
| **Router–Specialist** | **0.811**    | 0.663       | 0.644       | 0.358              |
| **Blackboard**        | 0.789        | 0.633       | 0.653       | 0.332              |
| **Planner–Executor**  | 0.789        | 0.620       | 0.653       | 0.349              |
| **Deterministic**     | 0.778        | **0.716**   | 0.631       | **0.364**          |
| **Hierarchical**      | 0.697        | 0.588       | 0.676       | 0.344              |
| **ReAct**             | 0.611        | 0.599       | **0.682**   | 0.364              |

### Operational Metrics Performance

| Architecture          | Mean Latency (ms) | Input Tokens        | Output Tokens     | Cost (USD/run) |
| :-------------------- | :---------------- | :------------------ | :---------------- | :------------- |
| **Blackboard**        | 91,429 ± 19,646   | 3,004 ± 216         | 498 ± 145         | **0.00037**    |
| **Planner–Executor**  | 84,255 ± 17,190   | 3,219 ± 261         | 580 ± 160         | 0.00040        |
| **Deterministic**     | 93,020 ± 17,599   | 2,988 ± 231         | 528 ± 160         | 0.00038        |
| **Router–Specialist** | 98,913 ± 18,361   | 3,199 ± 247         | 519 ± 152         | 0.00039        |
| **Hierarchical**      | 147,096 ± 68,594  | 450,017 ± 1,080,756 | 115,761 ± 303,962 | 0.07109        |
| **ReAct**             | 181,936 ± 45,116  | 391,935 ± 419,373   | 58,548 ± 66,709   | 0.04320        |

---

## Failure Analysis

Analysis of poorly scoring runs identified three prominent architectural failure modes:

### Profile 1 — Incomplete Answers (Deterministic, Blackboard)

The generated response is factually grounded but omits vital contractual exclusions. The root cause is _vector space polarity asymmetry_: affirmative user inquiries map closely to affirmative coverage clauses, failing to retrieve negating exclusion clauses which reside in a separate vector space region.

### Profile 2 — Overextended Answers (Router–Specialist, Planner–Executor)

The core response is accurate but supplemented with unverified details fetched from adjacent text chunks. While the target clauses are retrieved correctly, the synthesis model incorporates peripheral text simply because it fell inside the retrieved context window (_over-extension_).

### Profile 3 — Mixed / Inconsistent Answers (Hierarchical, ReAct)

Partial answers from completely different products or insurance providers are merged into a single text output. In the hierarchical setup, unconditional decomposition forces independent retrieval calls that pull data from unrelated policies; the final synthesizer merges them without cross-checking the product scope. In the ReAct topology, iterative query rewriting induces semantic drift, leading to cross-contamination of context.

---

## Practitioner Recommendations

1. **Adopt a Deterministic Baseline:** Use the deterministic architecture as the primary deployment target. It provides the highest baseline correctness (0.716), guarantees predictability, and defaults to omissions rather than hallucinated contradictions—a safer failure mode in regulated settings.
2. **Deploy Routing Conditionally:** The router–specialist architecture should be integrated only if the production distribution contains a significant volume of highly complex questions (+0.111 faithfulness gain on complex queries at negligible cost).
3. **Enforce Document Scope Separation:** Maintain separate Qdrant collections or explicit categorical metadata filters for OWU and IPID data. Failing to isolate them leads to authority conflicts where the vector search prioritizes high-level summarized charts (IPID) over binding contractual text (OWU).
4. **Structure-Aligned Chunking:** Align chunk boundaries with legal structure markers (sections, paragraphs) at roughly 256 tokens to prevent semantic bleeding of unrelated legal provisions into a single vector chunk.
5. **Role-Based Exclusion Indexing:** Inject a `role` metadata attribute (e.g., _coverage_, _exclusion_, _definition_) and programmatically guarantee the retrieval of at least one exclusion chunk for every coverage question to mitigate Profile 1 issues.

---

## Technology Stack

- **Agent Orchestration:** LangGraph v0.2.0
- **LLM Tooling Framework:** LangChain & `langchain-openai` v1.2.10
- **Inference Engine:** OpenAI API (`gpt-5`, `text-embedding-3-large`) v2.21.0
- **Evaluation & Observability:** Arize Phoenix v13 & OpenTelemetry auto-instrumentation
- **Databases:** Qdrant v1.15 (Vector Index), PostgreSQL v14 (Phoenix backend state store)
- **Data Validation:** Pydantic v2.12.5
- **PDF Extraction:** `pypdfium2` & Pillow v5.4.0

---

## Repository Structure

```
master-thesis/
├── README.md                          # Main repository documentation
├── thesis/                            # LaTeX thesis documentation sources
│   ├── thesis.tex                     # Primary compilation entry point
│   ├── references.bib                 # BibTeX bibliography database
│   └── thesis.pdf                     # Compiled academic manuscript
└── code/                              # Experimental framework implementation
    ├── sources/                       # Application source package
    │   ├── agents/                    # LangGraph workflow definitions
    │   │   ├── factory.py             # Graph building orchestration entry point
    │   │   ├── state.py               # ExperimentState schema definition
    │   │   ├── deterministic.py       # Linear DAG pipeline implementation
    │   │   └── ...                    # Remaining agent architectures
    │   ├── tools/                     # 8 asynchronous tool coroutines
    │   ├── config/                    # Prompt repositories, schemas, configuration
    │   ├── runner.py                  # Concurrent experiment processing manager
    │   └── evaluator.py               # Arize Phoenix verification interface
    ├── notebooks/                     # Analytical Jupyter Notebooks (NB_00 - NB_04)
    ├── data/                          # Source PDFs, evaluation items, and results
    ├── docker-compose.yml             # Orchestration for Qdrant, Phoenix, and Postgres
    └── pyproject.toml                 # uv dependency management configuration
```

---

## Shared Tools

Every tool is an asynchronous coroutine adhering to the following interface:  
`async def tool_name(state: ExperimentState, *, config: GraphConfig, **kwargs) -> dict:`

- `question_parser`: Extracts structured query metadata (company, insurance product, intent class).
- `query_rewriter`: Yields alternative search queries (Direct, Step-back, HyDE).
- `retriever`: Queries Qdrant vectors with adaptive metadata filter relaxation techniques.
- `reranker`: Implements LLM-driven relevance re-scoring (disabled during the baseline experiment).
- `evidence_selector`: Selects top-k diverse chunks using a deterministic pruning rule.
- `citation_maker`: Builds structured, traceable citation objects from selected context chunks.
- `prompt_selector`: Maps the extracted intent to one of 11 domain-specific generation templates.
- `answer_synthesizer`: Synthesizes the finalized Polish response using the mapped context.

---

## Notebooks

Notebooks are designed to be fully idempotent, automatically bypassing expensive API/processing operations if target files are found.

- `NB_00_visualize_agents.ipynb`: Renders Mermaid architecture layout definitions into SVG/PDF (< 1 min).
- `NB_01_build_vectorstore.ipynb`: Manages vision-based PDF OCR extraction, chunking, and Qdrant population (2-3 h).
- `NB_02_run_agents.ipynb`: Executes the parallelized multi-agent experiment suite ( 2 h across 32 threads).
- `NB_03_evaluate_agents.ipynb`: Coordinates the automated Phoenix LLM-as-judge evaluation process ( 1 h).
- `NB_04_visualize_results.ipynb`: Yields statistical data plots, matrices, and performance summaries (< 1 min).

---

## Setup and Execution

### Prerequisites

- Python 3.13 or newer
- Docker engine with Docker Compose installed
- Active OpenAI API credentials for the `gpt-5` model generation tier

### Step 1: Clone the Repository

```bash
git clone https://github.com/mchojna/master-thesis-pjatk.git
cd master-thesis-pjatk/code
```

### Step 2: Initialize Core Infrastructure

```bash
docker compose up -d
```

_Launches Qdrant (port 6333), Arize Phoenix (port 6006), and PostgreSQL (port 5432)._

### Step 3: Synchronize Local Environment

```bash
uv sync
```

### Step 4: Configure Local Parameters

Create an environment definition file named `.env` in the `code/` repository root:

```env
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
PHOENIX_HOST=localhost
PHOENIX_PORT=6006
POSTGRES_URL=postgresql://phoenix:phoenix@localhost:5432/phoenix
```

### Step 5: Execute Evaluation Pipeline

Launch the local notebook workspace: `jupyter notebook` and run notebooks `NB_01` through `NB_04` in linear succession.

### Executing a Single Graph Workflow Programmatically

```python
import asyncio
from sources.agents.factory import build_graph
from sources.config.graph import GraphConfig


async def main():
    config = GraphConfig(pattern_name="deterministic")
    graph = build_graph(config)
    result = await graph.ainvoke({
        "question": "Ile wariantów ubezpieczenia oferuje ERGO Podróż?",
        "pattern_name": "deterministic",
    })
    print(result["answer_with_references"])


asyncio.run(main())
```

---

## Extending the System

### Injecting a New Custom Architecture Topology

1. Construct a new module at `code/sources/agents/my_architecture.py` and supply a `build_graph(config: GraphConfig)` method returning an initialized `StateGraph`.
2. Add your architecture pointer definition inside `ALL_PATTERNS` found in `sources/agents/factory.py`.

### Registering a New Tool Coroutine

1. Provide an async function definition inside the `code/sources/tools/my_tool.py` workspace.
2. Bind the new reference and its structural definition within `sources/tools/__init__.py`:

   ```python
   from .my_tool import my_tool

   TOOL_REGISTRY["my_tool"] = my_tool
   TOOL_DESCRIPTIONS["my_tool"] = (
       "Description metadata utilized during execution planning tasks."
   )
   ```

---

## Thesis Structure

The compiled academic manuscript file is available directly at `thesis/thesis.pdf`. Recompilation from source requires TeX Live 2024+ and is triggered via: `cd thesis && latexmk -pdf thesis.tex`.

- **Chapter 1: Introduction:** Research landscape motivation, core questions (RQ1–RQ4), goals (RG1–RG4), and architectural scope definitions.
- **Chapter 2: Theoretical Foundations:** Multi-agent systems design patterns, advanced RAG systems, structural syntax challenges of insurance prose, and domain literature review.
- **Chapter 3: Methodology:** Controlled comparative experiment design constraints, dataset taxonomy rules, and formulation of performance metrics.
- **Chapter 4: Implementation Framework:** System architecture design, translation of graph logic using LangGraph structures, and analysis of custom core tool behaviors.
- **Chapter 5: Evaluation and Results:** Granular presentation of collected empirical trial benchmarks, statistical variance tests, and cost efficiency considerations.
- **Chapter 6: Conclusions:** Resolutions to original research questions, known systemic boundaries, and recommendations for future production systems architectures.

---

_Source code is released under the MIT License. Thesis text is copyright 2026 Michał Chojna. All rights reserved. Insurance documents utilized in this research remain the property of their respective issuers and were employed exclusively for non-commercial academic research purposes._
