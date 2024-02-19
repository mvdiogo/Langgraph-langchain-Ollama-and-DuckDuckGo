import pprint
from typing import Dict, TypedDict
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma

local_llm = 'mistral:instruct'
url = 'https://advogadosdf.com/artigos/inventario/'
model_name = 'intfloat/multilingual-e5-large'
inputs = {
    'keys': {
        'question': 'Quais os tipos de inventários existem?',
        'local': 'True',
    }
}


def loadDocument (local_llm=local_llm,url=url ,model_name=model_name):
    print("---LOAD DOCUMENT---")
    loader = WebBaseLoader(url)
    loader.default_parser = (
        'html.parser'  # html.parser, lxml, xml, lxml-xml, html5lib.
    )
    docs = loader.load()
    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800, chunk_overlap=200
    )
    all_splits = text_splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name=model_name)

    # Index
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        collection_name='rag-chroma',
        embedding=embedding,
    )
    retriever = vectorstore.as_retriever()
    return retriever


class GraphState(TypedDict):
    keys: Dict[str, any]

def retrieve(state):
    retriever = loadDocument()
    print('---RETRIEVE---')
    state_dict = state['keys']
    question = state_dict['question']
    local = state_dict['local']
    documents = retriever.get_relevant_documents(question)
    return {
        'keys': {'documents': documents, 'local': local, 'question': question}
    }


def generate(state):
    print('---GENERATE---')
    state_dict = state['keys']
    question = state_dict['question']
    documents = state_dict['documents']
    local = state_dict['local']

    template = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question}\nContext: {context}\nAnswer:"
    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template=template,
    )

    llm = ChatOllama(model=local_llm, temperature=0)

    rag_chain = prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({'context': documents, 'question': question})
    return {
        'keys': {
            'documents': documents,
            'question': question,
            'generation': generation,
        }
    }


def grade_documents(state):
    print('---CHECK RELEVANCE---')
    state_dict = state['keys']
    question = state_dict['question']
    documents = state_dict['documents']
    local = state_dict['local']

    llm = ChatOllama(model=local_llm, format='json', temperature=0)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.""",
        input_variables=['question', 'context'],
    )

    chain = prompt | llm | JsonOutputParser()

    # Score
    filtered_docs = []
    search = 'No'  # Default do not opt for web search to supplement retrieval
    for d in documents:
        score = chain.invoke(
            {
                'question': question,
                'context': d.page_content,
            }
        )
        grade = score['score']
        if grade == 'yes':
            print('---GRADE: DOCUMENT RELEVANT---')
            filtered_docs.append(d)
        else:
            print('---GRADE: DOCUMENT NOT RELEVANT---')
            search = 'Yes'  # Perform web search
            continue

    return {
        'keys': {
            'documents': filtered_docs,
            'question': question,
            'local': local,
            'run_web_search': search,
        }
    }


def transform_query(state):
    print('---TRANSFORM QUERY---')
    state_dict = state['keys']
    question = state_dict['question']
    documents = state_dict['documents']
    local = state_dict['local']

    # Create a prompt template with format instructions and the query
    prompt = PromptTemplate(
        template="""You are generating questions that is well optimized for retrieval. \n 
        Look at the input and try to reason about the underlying sematic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Provide an improved question without any premable, only respond with the updated question: """,
        input_variables=['question'],
    )

    # Grader
    llm = ChatOllama(model=local_llm, temperature=0)

    # Prompt
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({'question': question})

    return {
        'keys': {
            'documents': documents,
            'question': 'better - ' + better_question,
            'local': local,
        }
    }


def web_search(state):
    print('---WEB SEARCH---')
    try:
        state_dict = state['keys']
        question = state_dict['question']
        documents = state_dict['documents']
        local = state_dict['local']
        tool = DuckDuckGoSearchResults()
        docs = tool.invoke({'query': question})
        docs = extract_list_from_string(docs)
        web_results = '\n'.join(
            [
                d['content'] if isinstance(d, dict) and 'content' in d else d
                for d in docs
            ]
        )
        web_results = Document(page_content=web_results)
        documents.append(web_results)

        return {
            'keys': {'documents': documents, 'local': local, 'question': question}
        }
    except Exception as e:
        print(e)


def extract_list_from_string(input_string):
    # Extract content between square brackets, remove quotes, and split into a list
    try:
        result = (
            input_string.split('[')[1]
            .split(']')[0]
            .replace('"', '')
            .split(', ')
        )
    except Exception as e:
        print(e)
        result = input_string
    return result

### Edges

def decide_to_generate(state):
    print('---DECIDE TO GENERATE---')
    state_dict = state['keys']
    question = state_dict['question']
    filtered_documents = state_dict['documents']
    search = state_dict['run_web_search']

    if search == 'Yes':
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print('---DECISION: TRANSFORM QUERY and RUN WEB SEARCH---')
        return 'transform_query'
    else:
        # We have relevant documents, so generate answer
        print('---DECISION: GENERATE---')
        return 'generate'


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node('retrieve', retrieve)  # retrieve
workflow.add_node('grade_documents', grade_documents)  # grade documents
workflow.add_node('generate', generate)  # generatae
workflow.add_node('transform_query', transform_query)  # transform_query
workflow.add_node('web_search', web_search)  # web search


# Build graph
workflow.set_entry_point('retrieve')
workflow.add_edge('retrieve', 'grade_documents')
workflow.add_conditional_edges(
    'grade_documents',
    decide_to_generate,
    {
        'transform_query': 'transform_query',
        'generate': 'generate',
    },
)
workflow.add_edge('transform_query', 'web_search')
workflow.add_edge('web_search', 'generate')
workflow.add_edge('generate', END)

# Compile
app = workflow.compile()

# Run

for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint.pprint(f"Node '{key}':")
        # Optional: print full state at each node
        pprint.pprint(value['keys'], indent=2, width=80, depth=None)
    pprint.pprint('\n---\n')

# Final generation
pprint.pprint(value['keys']['generation'])
