from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings , ChatHuggingFace , HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate



# Step 1a - Indexing (Document Ingestion)

video_id = "Gfr50f6ZBvo" # only the ID, not full URL
try:
    # If you don’t care which language, this returns the “best” one
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id, languages=["en"])

    # Flatten it to plain text
    transcript = " ".join(chunk.text for chunk in transcript_list)
    # print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")
    
# Step 1b - Indexing (Text Splitting)
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
chunks = splitter.create_documents([transcript])

# Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks,embeddings)


# Step 2 - Retrieval

retriever = vector_store.as_retriever(search_type="mmr",search_kwargs={"k":8, "fetch_k":30})

#Step 3 - Argumentation
llm = HuggingFaceEndpoint(repo_id='Qwen/Qwen2.5-7B-Instruct')
model = ChatHuggingFace(llm=llm)


prompt = PromptTemplate(
    template="""Your are a helpful assistance.
    answer ONLY from the provided transcript context.
    if the context is insufficient , just say you dont know.
    
    {context}
    Question:{question}
    """,
    input_variables=['context', 'question']
)

# question = "Is the topic of fusion discussed in this video ? if yes then what was discussed"
# retrieved_docs = retriever.invoke()


# context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)


# final_prompt = prompt.invoke({'context':context_text, "question":question})


# Generation 

# answer = model.invoke(final_prompt)
# print(answer.content)


# Building chain
from langchain_core.runnables import RunnableParallel ,  RunnablePassthrough , RunnableLambda
from langchain_core.output_parsers import StrOutputParser


def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})


parser = StrOutputParser()

main_chain = parallel_chain | prompt | model | parser 

print(main_chain.invoke('Can you summarize the video'))