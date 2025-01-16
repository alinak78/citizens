import pandas as pd
import nltk
from sklearn.decomposition import LatentDirichletAllocation
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim
from wordcloud import WordCloud
from io import BytesIO
import base64
from collections import Counter

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import stopwords
from nltk.util import ngrams

# Load the CSV file
df = pd.read_csv("CitizensNYC_Project_Summaries.csv")
organization_names = df['Organization/Group Name: Organization Purpose and History'].dropna().tolist()
project_descript = df['Project Description'].dropna().tolist()
project_timeline = df['Project Timeline Summary'].dropna().tolist()


# Custom stopwords
#custom_stopwords = set(stopwords.words('english')).union({"climate", "solution", "adaptation", "solutions", "project"})

# Preprocess text function (noun-noun bigrams only)
def preprocess_text_with_noun_noun_bigrams(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    #tokens = [word for word in tokens if word not in custom_stopwords]
    
    # POS tagging
    pos_tags = nltk.pos_tag(tokens)
    
    # Extract noun-noun bigrams
    bigrams = [
        f"{word1} {word2}"
        for (word1, tag1), (word2, tag2) in ngrams(pos_tags, 2)
        if tag1.startswith('NN') and tag2.startswith('NN')
    ]
    return bigrams

# Preprocess all descriptions to noun-noun bigrams

texts_org = [preprocess_text_with_noun_noun_bigrams(text) for text in organization_names]
texts_project = [preprocess_text_with_noun_noun_bigrams(text) for text in project_descript]
texts_timeline = [preprocess_text_with_noun_noun_bigrams(text) for text in project_timeline]

# Flatten the bigrams for frequency analysis
all_bigrams_org = [bigram for text in texts_org for bigram in text]
all_bigrams_proj = [bigram for text in texts_project for bigram in text]
all_bigrams_timeline = [bigram for text in texts_timeline for bigram in text]


# Analyze most common bigrams for additional stopwords
bigram_counts_org = Counter(all_bigrams_org)
bigram_counts_proj = Counter(all_bigrams_proj)
bigram_counts_time = Counter(all_bigrams_timeline)

print("Most common noun-noun bigrams:", bigram_counts_org.most_common(20))

# Gensim LDA for topic modeling (with noun-noun bigrams)
def gensim_lda(texts, num_topics=5, passes=15):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes, random_state=42)
    return lda_model, corpus, dictionary

lda_gensim1, corpus1, dictionary1 = gensim_lda(texts_org)
lda_gensim2, corpus2, dictionary2 = gensim_lda(texts_project)
lda_gensim3, corpus3, dictionary3 = gensim_lda(texts_timeline)

# Save Gensim LDA visualization as HTML
lda_display1 = pyLDAvis.gensim.prepare(lda_gensim1, corpus1, dictionary1)
lda_display2 = pyLDAvis.gensim.prepare(lda_gensim2, corpus2, dictionary2)
lda_display3 = pyLDAvis.gensim.prepare(lda_gensim3, corpus3, dictionary3)

pyLDAvis_html1 = pyLDAvis.prepared_data_to_html(lda_display1)
pyLDAvis_html2 = pyLDAvis.prepared_data_to_html(lda_display2)
pyLDAvis_html3 = pyLDAvis.prepared_data_to_html(lda_display3)

# Extract topic-term distributions with saliency filtering
def get_prevalent_noun_noun_bigrams_with_saliency(lda_model, dictionary, num_words=8, saliency_threshold=0.2):
    topics = lda_model.get_topics()  # Topic-word probabilities
    num_topics = topics.shape[0]
    bigram_saliency = {}

    # Calculate saliency for each bigram
    for word_id in range(topics.shape[1]):
        total_weight = topics[:, word_id].sum()
        for topic_idx in range(num_topics):
            word = dictionary[word_id]
            saliency = topics[topic_idx, word_id] / total_weight
            bigram_saliency[word] = max(bigram_saliency.get(word, 0), saliency)

    # Filter bigrams by saliency
    prevalent_bigrams = {}
    for topic_idx in range(num_topics):
        topic_bigrams = {
            dictionary[word_id]: topics[topic_idx, word_id]
            for word_id in range(topics.shape[1])
            if bigram_saliency[dictionary[word_id]] >= saliency_threshold
        }
        # Keep only the top `num_words` bigrams for each topic
        prevalent_bigrams[f"Topic {topic_idx + 1}"] = dict(
            sorted(topic_bigrams.items(), key=lambda x: x[1], reverse=True)[:num_words]
        )

    return prevalent_bigrams

# Generate inline HTML word clouds for noun-noun bigrams
def generate_html_wordclouds_bigrams(topic_bigrams, grid_columns=3):

    html_content = '<div style="display: flex; flex-wrap: wrap; gap: 20px;">'

    for idx, (topic, bigrams) in enumerate(topic_bigrams.items(), start=1):
        # Generate the word cloud
        wordcloud = WordCloud(width=400, height=300, background_color="white").generate_from_frequencies(bigrams)
        
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

# Generate prevalent noun-noun bigrams filtered by saliency and visualize as inline HTML word clouds
prevalent_bigrams1 = get_prevalent_noun_noun_bigrams_with_saliency(lda_gensim1, dictionary1, num_words=15, saliency_threshold=0.3)
wordclouds_html1 = generate_html_wordclouds_bigrams(prevalent_bigrams1, grid_columns=3)

prevalent_bigrams2 = get_prevalent_noun_noun_bigrams_with_saliency(lda_gensim2, dictionary2, num_words=15, saliency_threshold=0.3)
wordclouds_html2 = generate_html_wordclouds_bigrams(prevalent_bigrams2, grid_columns=3)

prevalent_bigrams3 = get_prevalent_noun_noun_bigrams_with_saliency(lda_gensim3, dictionary3, num_words=15, saliency_threshold=0.3)
wordclouds_html3 = generate_html_wordclouds_bigrams(prevalent_bigrams3, grid_columns=3)




# Combine PyLDAvis HTML and Word Clouds HTML
final_1 = f"""
<html>
<body>
    {prevalent_bigrams1}
    <hr>
    {wordclouds_html1}
</body>
</html>
"""

final_2 = f"""
<html>
<head>
    <title>LDA Visualization for Project Descriptions</title>
</head>
<body>
    {prevalent_bigrams2}
    <hr>
    {wordclouds_html2}
</body>
</html>
"""

final_3 = f"""
<html>
<head>
    <title>LDA Visualization with Project Timeline summary</title>
</head>
<body>
    {prevalent_bigrams3}
    <hr>
    {wordclouds_html3}
</body>
</html>
"""

# Save the combined HTML file
output_file1 = "lda_visualization_org_name.html"
with open(output_file1, "w") as f:
    f.write(final_1)


output_file2 = "lda_visualization_project_descr.html"
with open(output_file2, "w") as f:
    f.write(final_2)

output_file3 = "lda_visualization_project_timeline.html"
with open(output_file3, "w") as f:
    f.write(final_3)


html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LDA Visualization</title>
    <style>
        .ldaContent {{ display: none; }}
    </style>
</head>
<body>
    <select id="sourceSelector" onchange="updateVisualization()">
        <option value="organization">Project Organization Names</option>
        <option value="description">Project Descriptions</option>
        <option value="timeline">Project Timeline Summary</option>
    </select>
    <div id="ldaContainer">
        <div id="organization" class="ldaContent">
        <h1>LDA Visualization of prevalent bigrams for Organization Names and History</h1>
        {pyLDAvis_html1}
        <h1>Wordcloud Visualization of prevalent bigrams for Organization Names and History</h1>
        {wordclouds_html1}
        </div>

        <div id="description" class="ldaContent">
        <h1>LDA Visualization of prevalent bigrams for Project Descriptions</h1>
        {pyLDAvis_html2}
        <h1>Wordcloud Visualization of prevalent bigrams for Project Descriptions</h1>
        {wordclouds_html2}
        </div>

        <div id="timeline" class="ldaContent">
        <h1>LDA Visualization of prevalent bigrams for Project Timeline</h1>
        {pyLDAvis_html3}
        <h1>Wordcloud Visualization of prevalent bigrams for Project Timeline</h1>
        {wordclouds_html3}
        </div>
    </div>
    
    <script>
        function updateVisualization() {{
            const selectedSource = document.getElementById('sourceSelector').value;
            document.querySelectorAll('.ldaContent').forEach(div => {{
                div.style.display = 'none';
            }});
            document.getElementById(selectedSource).style.display = 'block';
        }}
        // Show the first visualization by default
        document.getElementById('organization').style.display = 'block';
    </script>
</body>
</html>
"""

# Save the combined HTML file
output_file = "lda_visualization_dropdown.html"
with open(output_file, "w") as f:
    f.write(html_template)

print(f"LDA visualization with dropdown saved as '{output_file}'")