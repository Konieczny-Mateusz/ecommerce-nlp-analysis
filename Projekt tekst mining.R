# ==============================================================================
# 0. PRZYGOTOWANIE ŚRODOWISKA I WCZYTANIE DANYCH
# ==============================================================================
# install.packages(c("tidyverse", "tidytext", "tm", "wordcloud2", "wordcloud", "topicmodels", "igraph", "ggraph", "reshape2", "textdata"))

library(tidyverse)
library(tidytext)
library(tm)
library(wordcloud2)
library(wordcloud)    
library(topicmodels)
library(igraph)
library(ggraph)
library(reshape2)     
library(textdata)     # Wymagane do pobrania leksykonu NRC

# Wczytanie danych
df <- read_csv("Womens Clothing E-Commerce Reviews.csv")

# Globalne wczytanie słowników (Bing i NRC)
bing_lexicon <- get_sentiments("bing")
nrc_lexicon <- get_sentiments("nrc") %>% 
  filter(!sentiment %in% c("positive", "negative")) # Bierzemy tylko 8 konkretnych emocji

# Czyszczenie danych bazowych i zachowanie nowych zmiennych (Age)
df_clean <- df %>%
  filter(!is.na(`Review Text`)) %>%
  mutate(text = paste(Title, `Review Text`, sep = " ")) %>%
  mutate(doc_id = row_number()) %>%
  # Liczymy słowa w surowym tekście przed wyrzuceniem stop-words
  mutate(word_count = str_count(text, "\\w+")) %>%
  select(doc_id, text, Rating, Department = `Department Name`, Age, word_count)

# ==============================================================================
# 1. CZYSZCZENIE TEKSTU I TOKENIZACJA (NLP PREPROCESSING)
# ==============================================================================
tokens <- df_clean %>%
  mutate(text = str_replace_all(text, "[^a-zA-Z\\s]", " ")) %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words, by = "word") %>%
  filter(str_length(word) > 2)

# ==============================================================================
# 2. PODSTAWOWA ANALIZA TEKSTU I WIZUALIZACJE
# ==============================================================================
# A. TF-IDF dla kategorii Sukienki (Dresses)
tfidf_words <- tokens %>%
  count(Department, word, sort = TRUE) %>%
  filter(!is.na(Department)) %>%
  bind_tf_idf(word, Department, n)

tfidf_words %>%
  filter(Department == "Dresses") %>%
  top_n(10, tf_idf) %>%
  ggplot(aes(reorder(word, tf_idf), tf_idf)) +
  geom_col(fill = "coral") +
  coord_flip() +
  labs(title = "Słowa charakterystyczne dla kategorii Sukienki (Waga TF-IDF)", x = NULL, y = "Wartość TF-IDF") +
  theme_minimal()

# B. Analiza Bigramów i Wykres Sieciowy
bigrams <- df_clean %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(!word1 %in% stop_words$word, !word2 %in% stop_words$word) %>%
  count(word1, word2, sort = TRUE) %>%
  filter(n > 100)

graph <- graph_from_data_frame(bigrams)
ggraph(graph, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE, color = "black") +
  geom_node_point(color = "darkorange", size = 4) +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
  theme_void() +
  labs(title = "Sieć powiązań między słowami (Bigramy)")

# C. Podstawowa chmura słów
word_counts <- tokens %>% count(word, sort = TRUE)
set.seed(123)
wordcloud(words = word_counts$word, freq = word_counts$n, min.freq = 50,
          max.words = 150, random.order = FALSE, rot.per = 0.35, 
          colors = brewer.pal(8, "Dark2"))

# ==============================================================================
# 3. ZAAWANSOWANA ANALIZA SENTYMENTU (BING - TYLKO POZYTYWNE/NEGATYWNE)
# ==============================================================================
# Weryfikacja NLP: Sentyment a Rzeczywista Ocena (Podwójny wykres)
sentiment_analysis_both <- tokens %>%
  inner_join(bing_lexicon, by = "word", relationship = "many-to-many") %>%
  count(Rating, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(
    Pozytywne = positive / (positive + negative),
    Negatywne = negative / (positive + negative)
  ) %>%
  pivot_longer(cols = c(Pozytywne, Negatywne), 
               names_to = "Typ_sentymentu", 
               values_to = "Wskaznik")

ggplot(sentiment_analysis_both, aes(x = factor(Rating), y = Wskaznik, fill = Typ_sentymentu)) +
  geom_col(position = "dodge", color = "white") +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_manual(values = c("Pozytywne" = "#2ca25f", "Negatywne" = "#de2d26")) +
  labs(title = "Struktura emocjonalna opinii w zależności od oceny produktu", 
       subtitle = "Odsetek słów pozytywnych vs negatywnych",
       x = "Ocena Gwiazdkowa (1-5)", y = "Procent słów", fill = "Wydźwięk:") +
  theme_minimal() + theme(legend.position = "top")

# ==============================================================================
# 4. MODELOWANIE TEMATÓW (LDA - Latent Dirichlet Allocation)
# ==============================================================================
set.seed(123)
sample_docs <- sample(df_clean$doc_id, 2000)

dtm <- tokens %>%
  filter(doc_id %in% sample_docs) %>%
  count(doc_id, word) %>%
  cast_dtm(doc_id, word, n)

lda_model <- LDA(dtm, k = 3, control = list(seed = 1234))
tematy <- tidy(lda_model, matrix = "beta")

tematy_z_sentymentem <- tematy %>%
  # 1. Używamy left_join, aby zachować słowa, których nie ma w słowniku
  left_join(bing_lexicon, by = c("term" = "word"), relationship = "many-to-many") %>%
  # 2. Brakujące przypisania zamieniamy na "neutralny"
  mutate(sentiment = replace_na(sentiment, "neutralny")) %>%
  mutate(nazwa_kategorii = case_when(
    topic == 1 ~ "Rozmiar i Krój",
    topic == 2 ~ "Materiał i Komfort",
    topic == 3 ~ "Styl i Wygląd"
  )) %>%
  # 3. Wybieramy po 5 słów dla każdego tematu i dla każdej z TRZECH kategorii emocjonalnych
  group_by(nazwa_kategorii, sentiment) %>%
  top_n(5, beta) %>%
  ungroup() %>%
  arrange(nazwa_kategorii, -beta)

tematy_z_sentymentem %>%
  ggplot(aes(reorder_within(term, beta, nazwa_kategorii), beta, fill = sentiment)) +
  geom_col() +
  facet_wrap(~ nazwa_kategorii, scales = "free") +
  coord_flip() +
  scale_x_reordered() +
  # 4. Dodajemy kolor szary (gray70) dla słów neutralnych
  scale_fill_manual(values = c("positive" = "#2ca25f", "negative" = "#de2d26", "neutralny" = "gray70")) +
  labs(title = "Profil słownikowy ukrytych tematów (Model LDA)",
       subtitle = "Top 5 słów pozytywnych, negatywnych i neutralnych na każdy temat",
       x = "Słowo", y = "Prawdopodobieństwo (Beta)", fill = "Sentyment:") +
  theme_minimal() + 
  theme(legend.position = "bottom", strip.text = element_text(size = 12, face = "bold"))

# ==============================================================================
# 5. NOWE ANALIZY ZAAWANSOWANE (EMOCJE NRC, DEMOGRAFIA, DŁUGOŚĆ)
# ==============================================================================

# 1. Przygotowanie danych i tłumaczenie emocji na język polski
emotion_profile <- tokens %>%
  inner_join(nrc_lexicon, by = "word", relationship = "many-to-many") %>%
  filter(Rating %in% c(1, 5)) %>%
  count(Rating, sentiment) %>%
  # Tłumaczenie słownika NRC z angielskiego na polski dla lepszego odbioru
  mutate(sentiment_pl = case_when(
    sentiment == "anger" ~ "Gniew",
    sentiment == "anticipation" ~ "Oczekiwanie",
    sentiment == "disgust" ~ "Obrzydzenie",
    sentiment == "fear" ~ "Strach",
    sentiment == "joy" ~ "Radość",
    sentiment == "sadness" ~ "Smutek",
    sentiment == "surprise" ~ "Zaskoczenie",
    sentiment == "trust" ~ "Zaufanie",
    TRUE ~ sentiment
  )) %>%
  group_by(Rating) %>%
  mutate(proporcja = n / sum(n)) %>%
  ungroup() %>%
  # KLUCZOWY KROK: Wartości dla 1 gwiazdki robimy ujemne (żeby poszły na lewo osi)
  mutate(proporcja_wykres = ifelse(Rating == 1, -proporcja, proporcja))

# 2. Generowanie wykresu piramidowego (Diverging Bar Chart)
ggplot(emotion_profile, aes(x = reorder(sentiment_pl, proporcja_wykres), y = proporcja_wykres, fill = factor(Rating))) +
  geom_col(width = 0.7) +
  coord_flip() +
  # Linia zerowa (środek lustra)
  geom_hline(yintercept = 0, color = "black", linewidth = 0.8) +
  # Kolory: Czerwony dla 1 gwiazdki, Zielony dla 5 gwiazdek
  scale_fill_manual(values = c("1" = "#de2d26", "5" = "#2ca25f"), 
                    labels = c("1 Gwiazdka (Krytyka)", "5 Gwiazdek (Zachwyt)")) +
  # Funkcja abs() w labels sprawia, że na osi X nie ma "ujemnych" procentów, obie strony są na plusie
  scale_y_continuous(labels = function(x) scales::percent(abs(x)), 
                     limits = c(-max(abs(emotion_profile$proporcja_wykres)), 
                                max(abs(emotion_profile$proporcja_wykres)))) +
  labs(title = "Wykres Piramidowy: Profil Emocjonalny Opinii",
       subtitle = "Różnice w ładunku emocjonalnym między skrajnymi ocenami (Słownik NRC)",
       x = "Rodzaj emocji", 
       y = "Udział danej emocji w tekście", 
       fill = "Ocena produktu:") +
  theme_minimal() +
  theme(legend.position = "top", 
        axis.text.y = element_text(face = "bold", size = 11),
        plot.title = element_text(face = "bold", size = 14))

# 5.2 WPŁYW WIEKU KLIENTEK NA SENTYMENT (DEMOGRAFIA)
# 1. Przygotowanie danych (inner_join odsiewa słowa bez emocji)
age_sentiment <- tokens %>%
  inner_join(bing_lexicon, by = "word", relationship = "many-to-many") %>%
  count(doc_id, Age, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(
    # Tworzenie grup wiekowych
    Age_Group = cut(Age, breaks = c(0, 30, 40, 50, 60, 100), labels = c("<30", "30-40", "40-50", "50-60", "60+")),
    # Liczymy proporcje tylko na podstawie słów nacechowanych emocjonalnie
    Suma_emocji = positive + negative,
    Pozytywne = positive / Suma_emocji,
    Negatywne = negative / Suma_emocji
  ) %>%
  # Zabezpieczenie przed błędem dzielenia przez 0
  filter(Suma_emocji > 0) %>% 
  group_by(Age_Group) %>%
  # Uśrednianie wyników dla danej grupy
  summarise(
    Pozytywne = mean(Pozytywne, na.rm = TRUE),
    Negatywne = mean(Negatywne, na.rm = TRUE)
  ) %>%
  # Przejście do formatu "long" dla ggplot
  pivot_longer(cols = c(Pozytywne, Negatywne), 
               names_to = "Typ_sentymentu", 
               values_to = "Wskaznik")

# 2. Rysowanie wykresu z dwoma wskaźnikami
ggplot(age_sentiment, aes(x = Age_Group, y = Wskaznik, fill = Typ_sentymentu)) +
  geom_col(position = "dodge", color = "white") +
  # Etykiety tekstowe nad słupkami
  geom_text(aes(label = scales::percent(Wskaznik, accuracy = 0.1)), 
            position = position_dodge(width = 0.9), vjust = -0.5, size = 3.5, fontface = "bold") +
  # Kolorystyka binarnego podziału
  scale_fill_manual(values = c("Pozytywne" = "#2ca25f", "Negatywne" = "#de2d26")) +
  # Zapas na osi Y, żeby zmieściły się podpisy
  scale_y_continuous(labels = scales::percent, limits = c(0, max(age_sentiment$Wskaznik) * 1.15)) +
  labs(title = "Czy wiek wpływa na proporcje emocji w opinii?",
       subtitle = "Stosunek słów pozytywnych do negatywnych według grup wiekowych",
       x = "Grupa wiekowa", 
       y = "Udział w ładunku emocjonalnym", 
       fill = "Wydźwięk:") +
  theme_minimal() +
  theme(legend.position = "top", 
        plot.title = element_text(face = "bold", size = 14))

# 5.3 ANALIZA TRUDNOŚCI / DŁUGOŚCI TEKSTU (WORD COUNT VS RATING)
# Zależność długości napisanego tekstu od wystawionej oceny

ggplot(df_clean, aes(x = factor(Rating), y = word_count, fill = factor(Rating))) +
  # Wykres skrzypcowy (pokazuje gęstość danych)
  geom_violin(alpha = 0.4, color = NA, trim = FALSE) +
  # Mały boxplot wewnątrz skrzypiec pokazujący medianę i kwartyle
  geom_boxplot(width = 0.15, alpha = 0.8, outlier.shape = NA, color = "black") +
  # Dodanie kropki oznaczającej czerwoną średnią z poprzedniego wykresu
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "white", color = "darkred") +
  scale_fill_manual(values = c("#de2d26", "#fc9272", "#bdbdbd", "#a1d99b", "#31a354")) +
  # Ucięcie skrajnych outlierów, by wykres był zgrabny
  coord_cartesian(ylim = c(0, 150)) +
  labs(title = 'Rozkład długości tekstów a wystawiona ocena',
       subtitle = "Szerokość figury oznacza zagęszczenie opinii. Romb to wartość średnia.",
       x = "Ocena produktu (Rating)", 
       y = "Liczba słów (Word Count)") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 14),
        panel.grid.minor = element_blank())
