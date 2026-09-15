# 👗 E-commerce Reviews: NLP & Sentiment Analysis

Projekt analityczny wykorzystujący techniki Text Mining oraz Machine Learning do dogłębnej analizy recenzji klientek internetowego sklepu odzieżowego. Skrypt przekształca nieustrukturyzowany tekst opinii w mierzalne wskaźniki biznesowe i profile psychologiczne konsumentów.

## Zawartość projektu
* **Projekt-tekst-mining.html**: Gotowy raport analityczny (pobierz i otwórz w przeglądarce).
* **Projekt tekst mining.Rmd**: Plik R Markdown łączący kod z opisem metodologii.
* **Projekt tekst mining.R**: Czysty kod źródłowy w języku R.

## Główne funkcjonalności
* **Analiza Częstości (TF-IDF):** Identyfikacja słów charakterystycznych dla konkretnych kategorii ubrań (np. sukienek).
* **Modelowanie Tematów (LDA):** Nienadzorowane uczenie maszynowe grupujące recenzje w klastry tematyczne (Rozmiar/Krój, Materiał/Komfort, Styl/Wygląd).
* **Profilowanie Emocjonalne (Leksykon NRC):** Badanie nasycenia tekstu 8 bazowymi emocjami skontrastowane z oceną produktu na wykresach piramidowych.
* **Demografia i Behawiorystyka:** Analiza korelacji między wiekiem klientek a wydźwiękiem opinii oraz badanie zależności długości tekstu od liczby gwiazdek.

## Technologie
* **Język:** R
* **Biblioteki:** `tidyverse`, `tidytext`, `topicmodels` (LDA), `igraph`, `ggraph`, `wordcloud`.
