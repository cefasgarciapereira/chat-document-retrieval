from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from translate import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model
from transformers import AutoTokenizer, AutoModelForCausalLM
import nltk
import os

MODEL_PATH = "./models/llama-2-7b-chat.ggmlv3.q4_K_M.bin"

huggingfacehub_api_token="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"

# Pré-processamento dos documentos
nltk.download('punkt')
nltk.download('stopwords')

repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token, 
                     repo_id=repo_id, 
                     model_kwargs={"temperature":0.1, "max_new_tokens": 150})

# Supondo que você tenha um conjunto de documentos
documentos = [
    # Adicione mais documentos conforme necessário
]

folder = "./docs"

for filename in os.listdir(folder):
  f_content = open(folder+"/"+filename, "r").read()
  documentos.append(f_content)

# Função para pré-processar o texto
def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [w.lower() for w in tokens if w.isalnum()]  # Remover pontuações e converter para minúsculas
    tokens = [w for w in tokens if w not in nltk.corpus.stopwords.words('portuguese')]
    tokens = [w for w in tokens if w not in nltk.corpus.stopwords.words('english')]
    return ' '.join(tokens)

# Criar um vetorizador TF-IDF
vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
tfidf_matrix = vectorizer.fit_transform(documentos)

# Função para obter a resposta do chatbot
def get_context(pergunta):
    pergunta_tfidf = vectorizer.transform([pergunta])

    # Calcular a similaridade de cosseno entre a pergunta e os documentos
    similaridades = cosine_similarity(pergunta_tfidf, tfidf_matrix)

    # Encontrar o documento mais relevante
    documento_index = similaridades.argmax()

    # Retornar a resposta correspondente
    return documentos[documento_index]

def get_answer(context, question):
  template =  """
    Use the following context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    {context}
    Question: {question}
    Helpful Answer:
  """
  #prompt = PromptTemplate(template=template, input_variables=["context", "question"])
  prompt = PromptTemplate.from_template(template)
  llm_chain = LLMChain(prompt=prompt, llm=llm)
  response = llm_chain.run({"context": context, "question": question})
  return response

def make_a_question(question):
  context = get_context(question)
  answer = get_answer(context, question)
  return (answer, context)