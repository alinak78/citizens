import nltk
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
import base64
import json

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# Load the extracted answers from JSON
answers_file = "extracted_answers.json"  # Path to JSON with extracted answers
with open(answers_file, "r", encoding="utf-8") as f:
    answers_data = json.load(f)

# Combine all answers into a single text for processing
all_answers_text = " ".join(
    answer["Answer"] for person in answers_data.values() for answer in person.values()
)

# Split answers into sentences for processing
sections = nltk.sent_tokenize(all_answers_text)

# Preprocess text function
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens

# Preprocess the sections
texts = [preprocess_text(section) for section in sections if section.strip()]

# Gensim LDA for topic modeling
def gensim_lda(texts, num_topics=5, passes=15):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = gensim.models.LdaModel(
        corpus, num_topics=num_topics, id2word=dictionary, passes=passes, random_state=42
    )
    return lda_model, corpus, dictionary

lda_gensim, corpus, dictionary = gensim_lda(texts)

# Save Gensim LDA visualization as HTML
lda_display = pyLDAvis.gensim.prepare(lda_gensim, corpus, dictionary)
pyLDAvis_html = pyLDAvis.prepared_data_to_html(lda_display)

# Extract topic-term distributions with saliency filtering
def get_prevalent_words_with_saliency(lda_model, num_words=15, saliency_threshold=0.7):
    topics = lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False)
    all_topic_words = {}
    word_saliency = {}

    # Collect words and weights for each topic
    for topic_id, words in topics:
        topic_words = {word: weight for word, weight in words}
        all_topic_words[f"Topic {topic_id + 1}"] = topic_words

        # Compute overall saliency for each word
        for word, weight in topic_words.items():
            word_saliency[word] = word_saliency.get(word, 0) + weight

    # Filter words by saliency and uniqueness
    unique_topic_words = {}
    for topic, words in all_topic_words.items():
        unique_words = {
            word: weight
            for word, weight in words.items()
            if word_saliency[word] <= saliency_threshold
        }
        unique_topic_words[topic] = unique_words

    return unique_topic_words

# Generate inline HTML word clouds in a grid layout
def generate_html_wordclouds_grid(topic_words, grid_columns=3):
    html_content = "<h1>Word Clouds for LDA Topics (Filtered by Saliency)</h1>"
    html_content += '<div style="display: flex; flex-wrap: wrap; gap: 20px;">'

    for idx, (topic, words) in enumerate(topic_words.items(), start=1):
        # Generate the word cloud
        wordcloud = WordCloud(width=400, height=300, background_color="white").generate_from_frequencies(words)
        
        # Save the word cloud to a BytesIO object
        img_buffer = BytesIO()
        wordcloud.to_image().save(img_buffer, format="PNG")
        img_data = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        
        # Add the word cloud to the HTML
        html_content += f"""
        <div style="flex: 1 0 calc(33.33% - 20px); text-align: center; margin-bottom: 20px;">
            <h3>{topic}</h3>
            <img src="data:image/png;base64,{img_data}" alt="{topic}" style="max-width: 100%; height: auto;">
        </div>
        """

    html_content += "</div>"
    return html_content

# Generate prevalent words filtered by saliency and visualize as inline HTML word clouds
prevalent_words = get_prevalent_words_with_saliency(lda_gensim, num_words=15, saliency_threshold=0.2)
wordclouds_html = generate_html_wordclouds_grid(prevalent_words, grid_columns=3)

# Combine PyLDAvis HTML and Word Clouds HTML
final_html = f"""
<html>
<head>
    <title>Topic Modeling and Word Clouds</title>
</head>
<body>
    <h1>LDA Topic Modeling Visualization</h1>
    {pyLDAvis_html}
    <hr>
    {wordclouds_html}
</body>
</html>
"""

# Save the combined HTML file
output_file = "topics_and_wordclouds.html"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(final_html)

print(f"Final visualization saved as '{output_file}'")
