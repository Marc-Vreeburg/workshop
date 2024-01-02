# run deze code eenmalig om de benodigde software bibliotheken ('modules') te installeren
%pip install chromadb==0.4.18
%pip install openai==1.3.8
%pip install langchain==0.0.348
%pip install huggingface-hub==0.16.4
%pip install pypdf==3.17.2
%pip install openpyxl==3.0.10
%pip install docx2txt==0.8
%pip install unstructured[all-docs]==0.11.2
%pip install sentence-transformers==2.2.2
%pip install streamlit==1.29.0

# importeer de benodigde softwarebibliotheken ('modules')
import os
import streamlit as st
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain.document_loaders.excel import UnstructuredExcelLoader
from langchain.document_loaders.powerpoint import UnstructuredPowerPointLoader
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from huggingface_hub import hf_hub_download, login

# maak op uw harde schijf een mappenstructuur aan met de volgende mappen:
CURRENT_WORKING_DIRECTORY = os.getcwd() # NB gebruik GEEN mapnamen met spaties want dat resulteert in een onjuist pad en een foutmelding!!!
DOCUMENTEN_FOLDER_NAME = os.path.join(CURRENT_WORKING_DIRECTORY, 'documenten') # NB gebruik GEEN mapnamen met spaties want dat resulteert in een onjuist pad en een foutmelding!!!
EMBEDDINGS_FOLDER_NAME = os.path.join(CURRENT_WORKING_DIRECTORY, 'embeddingmodel') # NB gebruik GEEN mapnamen met spaties want dat resulteert in een onjuist pad en een foutmelding!!!
VECTORDB_FOLDER_NAME = os.path.join(CURRENT_WORKING_DIRECTORY, 'vectordb') # NB gebruik GEEN mapnamen met spaties want dat resulteert in een onjuist pad en een foutmelding!!!
#print(DOCUMENTEN_FOLDER_NAME)
#print(EMBEDDINGS_FOLDER_NAME)
#print(VECTORDB_FOLDER_NAME)

# Vul uw codes in voor de website van Huggingface en Openai
OPENAI_API_KEY = "GEEF HIER UW KEY IN"
HUGGINGFACE_API_KEY = "GEEF HIER UW KEY IN"

# Geef uw dossier een naam
NAME_COLLECTION = 'workshop_dossier'
# Geef de vector database een naam
NAME_VECTOR_DB = 'workshop'

def definieer_het_embedding_model():
    # definieer het voor het maken van de word embeddings te gebruiken model en download het zonodig van de website van Huggingface
    # voor RAG (Retrieval Augmented Generation) heeft u twee LLM's nodig: één voor het maken van de word embeddings/vectoren en één voor
    # de tekst interface (antwoorden op vragen)
    # voor het maken van word embeddings worden tekstfragmenten omgezet in tokens; commerciële llm's als chatgpt van openai berekenen kosten per verwerkte token
    # bij het maken van word embeddings gaat het al snel om relatief veel tokens en dus ook veel kosten
    # daarom maken we voor de llm voor het emdedden gebruik van een opensource model van de website van Huggingface
    # voor de tekst interface maken we gebruik van het commerciële chatgpt omdat dit een state-of-art model is en omdat het tijdens een chat om te zetten aantal tokens
    # relatief gering is en dus weinig kost
    emb_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=emb_model, cache_folder=EMBEDDINGS_FOLDER_NAME)
    login(token = HUGGINGFACE_API_KEY)
    return embeddings

def opslaan_documenten():
    # lees de documenten in de map 'DOCUMENTEN_FOLDER_NAME' in en knip deze op in tekstfragmenten ('chunks') zodat ze door de llm voor de tekst interface kunnen worden gebruikt
    for file in os.listdir(DOCUMENTEN_FOLDER_NAME):
        os.rename(os.path.join(DOCUMENTEN_FOLDER_NAME, file), os.path.join(DOCUMENTEN_FOLDER_NAME, file).replace(' ', '_').lower()) # NB gebruik GEEN mapnamen met spaties want dat resulteert in een onjuist pad en een foutmelding!!!
    loaders = {
            ".pdf": PyPDFLoader,
            ".csv": CSVLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".pptx": UnstructuredPowerPointLoader,
            ".txt": TextLoader
        }
    def maak_de_directory_loader(file_type, directory_path):
        return DirectoryLoader(path=directory_path, glob=f"**/*{file_type}", loader_cls=loaders[file_type], show_progress=True, use_multithreading=False, max_concurrency=1)
    pdf_loader = maak_de_directory_loader(".pdf", DOCUMENTEN_FOLDER_NAME)
    csv_loader = maak_de_directory_loader(".csv", DOCUMENTEN_FOLDER_NAME)
    docx_loader = maak_de_directory_loader(".docx", DOCUMENTEN_FOLDER_NAME)
    xlsx_loader = maak_de_directory_loader(".xlsx", DOCUMENTEN_FOLDER_NAME)
    pptx_loader = maak_de_directory_loader(".pptx", DOCUMENTEN_FOLDER_NAME)
    txt_loader = maak_de_directory_loader(".txt", DOCUMENTEN_FOLDER_NAME)
    documents = []
    for file in os.listdir(DOCUMENTEN_FOLDER_NAME):
        print(file)
        if file.endswith('.pdf'):
            documents.extend(pdf_loader.load())
        elif file.endswith(".csv"):
            documents.extend(csv_loader.load())
        elif file.endswith(".docx"):
            documents.extend(docx_loader.load())
        elif file.endswith(".xlsx"):
            documents.extend(xlsx_loader.load())
        elif file.endswith(".pptx"):
            documents.extend(pptx_loader.load())
        elif file.endswith(".txt"):
            documents.extend(txt_loader.load())
        elif file.endswith(".doc") or file.endswith(".xls") or file.endswith(".ppt"):
            print("")
            print("Document", file, "NIET verwerkt!!!")
        else:
            print("")
            print("Document", file, "NIET verwerkt!!!")
    # definieer een tekst splitter en knip de ingelezen documenten op in tekstfragmenten ('chunks')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=75, separators=["\n\n", "\n", " ", ""], length_function=len,) # 256 tokens max; bij 3 karakters per token dus ongeveer 750 karakters
    chunked_documents = text_splitter.split_documents(documents)
    print(len(chunked_documents))
    # definieer het voor het maken van de word embeddings te gebruiken model en download het zonodig van de website van Huggingface
    embeddings = definieer_het_embedding_model()
    # zet de tekstfragmenten (chunks) met behulp van word embeddings om in vectoren en sla deze vectoren op in de vector store 'Chroma'
    vectordb = Chroma.from_documents(documents=chunked_documents, embedding=embeddings, collection_name=NAME_COLLECTION, persist_directory=VECTORDB_FOLDER_NAME)
    vectordb.persist()
    return vectordb

def maken_qa_chain():
    # maak de pipeline voor het beantwoorden van vragen
    # kies een llm van openai, genaamd 'gpt-3.5-turbo', voor de tekstinterface
    # stel de query steeds samen ('stuff') uit 1. de vraag van de gebruiker en 2. de context (= uit de vector store opgehaalde meest bijpassende tekstfragmenten).
    # de llm zorgt voor een menselijke communicatie en de context zorgt voor de inbreng van kennis in de query
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY, temperature=0.3, max_tokens=2000)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

def verkrijgen_antwoord(query):
    # doorloop de stappen die nodig zijn om een antwoord te geven op een vraag
    # lees eerst de documenten in de map 'DOCUMENTEN_FOLDER_NAME' in
    # het resultaat is een gevulde vectore store
    vectordb = opslaan_documenten()
    # maak de pipeline voor het beantwoorden van vragen
    # het resultaat is een gedefinieerde pipeline
    chain = maken_qa_chain()
    # zoek in de vectore store de vectoren op met tekstfragmenten die het dichtst in de buurt komen bij de betekenis van de tekst in de vraag
    # zet daartoe de tekst in de vraag met behulp van word embedding om in een vector en zoek hier in de vectore store de meest vergelijkbare vectoren bij
    # selecteer de best 'k' vectoren uit de vector store, zet deze om in tekstfragmenten, voeg deze tekstfragmenten toe aan de tekst van de vraag en biedt
    # alles aan de eerder gemaakte pipeline aan.
    matching_docs = vectordb.similarity_search(query, k=4)
    # geef het aantal teksfragmenten weer
    #print(len(matching_docs))
    # geef het eerste teksfragment weer
    #print(matching_docs[0].page_content)
    # vraag het antwoord op
    answer = chain.run(input_documents=matching_docs, question=query)
    return answer

# Streamlit user interface
# maak een eenvoudige webserver en een webpagina voor de chat
# om de webpagina te zien/gebruiken gaat u als volgt te werk:
# 1. installeer streamlit (pip3 install streamlit)
# 2. open een windows shell door in de windows zoekbalk 'cmd' te typen, enter in te drukken en daarna op het icoontje dat verschijnt te klikken
# 3. type in het venster dat nu opent ' streamlit run "c:/het pad en de map waar dit script in staat/workshop_streamlit.py" '
# 4. in de browser zal nu een webpagina openen met de interface voor de app
st.set_page_config(page_title="Documenten zoeker", page_icon=":robot:")
st.header("Chat met uw pdf documenten")
form_input = st.text_input('Stel uw vraag')
submit = st.button("Verzend")
if submit:
    st.write(verkrijgen_antwoord(form_input))