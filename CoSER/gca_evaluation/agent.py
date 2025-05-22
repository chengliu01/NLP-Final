import json
from typing import Dict, List
from utils import config, fix_repeation
from openai import OpenAI
import copy
import json

from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS  
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

from utils import logger, config
from utils import get_character_prompt_w_cognitive_stage2
from cognitive_tools import extract_cognitive_analysis

ENVIRONMENT = "Environment"
NSP = "NSP"

special_characters = [ENVIRONMENT, NSP]

class Embedding(Embeddings):
    """A class that provides text embedding functionality using OpenAI library.
    
    This class implements the Embeddings interface and provides methods to:
    - Initialize an OpenAI client with proper credentials
    - Embed single pieces of text (queries)
    - Batch embed multiple texts (documents)
    - Handle errors and edge cases during the embedding process
    
    Attributes:
        client (OpenAI): OpenAI client instance for making API calls
        model (str): Name of the embedding model to use
        embedding_ctx_length (int): Maximum context length for embeddings
    """

    def __init__(self):
        # Initialize OpenAI client with credentials from config
        self.client = OpenAI(
            api_key=config['embedding_api_key'],
            base_url=config['embedding_base_url']
        )
        # Set the embedding model, default is typically "eval-BAAI-bge-m3-embedding"
        self.model = config['embedding_model']
        # Maximum tokens that can be embedded in one request
        self.embedding_ctx_length = 8192  

    def _embed(self, text: str) -> List[float]:
        """Internal method to embed a single piece of text.

        Args:
            text (str): The text to embed

        Returns:
            List[float]: The embedding vector

        Raises:
            ValueError: If input is not a string
            Exception: If embedding fails
        """
        if not isinstance(text, str):
            raise ValueError(f"Expected string input, got {type(text)}")
        
        # Replace newlines with spaces for cleaner input
        text = text.replace("\n", " ")
        try:
            embedding = self.client.embeddings.create(
                input=[text],
                model=self.model
            )
            return embedding.data[0].embedding
        except Exception as e:
            print(f"Error during embedding: {e}")
            print(f"Problematic text: {text[:100]}...")  
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a query text using the embedding model.

        Args:
            text (str): Query text to embed

        Returns:
            List[float]: Embedding vector for the query
        """
        return self._embed(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch embed multiple documents using the embedding model.

        Args:
            texts (List[str]): List of texts to embed

        Returns:
            List[List[float]]: List of embedding vectors

        Raises:
            Exception: If batch embedding fails
        """
        embeddings = []
        # Process texts in batches of 100 to avoid rate limits
        for i in range(0, len(texts), 100):  
            batch = texts[i:i+100]
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                embeddings.extend([data.embedding for data in response.data])
            except Exception as e:
                print(f"Error during batch embedding: {e}")
                raise
        return embeddings

def build_rag_corpus(character, database, target):
    """Build a retrieval-augmented generation (RAG) corpus from a character database.
    
    This function creates different types of document collections (book content, summaries, conversations)
    and builds vector stores for retrieval. It supports different retrieval targets and configurations.

    Args:
        character (str): Name of the character to build corpus for
        database (dict): Database containing character information, plots, conversations etc.
                        Should have 'detailed_plots' key with plot information.
        target (str): Type of corpus to build. Options:
            - 'raw_text': Only book content
            - 'expr1': Character and plot summaries
            - 'expr3': Summaries with k=3 retrieval
            - 'conv1': Only conversations
            - 'expr3_conv1': Mix of summaries and conversations
            - 'expr10_conv1': Expanded summaries with conversations

    Returns:
        dict: Mapping of corpus types to their respective retrievers.
              Returns None if database is None.
    """
    # Return None if no database provided
    if database is None:
        return None

    # Initialize empty corpora lists
    book_corpus = []
    experience_corpus = []
    conversation_corpus = [] 
    
    # Iterate through plots to build different corpora
    for plot in database['detailed_plots']:
        # Add full plot text to book corpus
        # Each plot is ~1000-3000 tokens, so no splitting needed
        book_corpus.append(plot['text'])

        # Build character-specific experience
        character_experience = { _['name']: _['experience'] for _ in  plot['key_characters']}.get(character, '')
        if character_experience:
            character_experience = f"{character}'s role: " + character_experience
        experience_corpus.append('PLOT: ' + plot['summary'] + '\n' + character_experience)

        # Process conversations, removing internal metadata
        conversation_info = {"summary": plot['summary'], "conversation": copy.deepcopy(plot['conversation'])}
        for conversation in conversation_info['conversation']:
            # Remove internal tracking fields from character info
            for character_info in conversation['key_characters']:
                character_info.pop("i_p", None)
                character_info.pop("i_c", None)
            
            # Remove internal tracking fields from dialogues
            for dialogue in conversation['dialogues']:
                dialogue.pop("i_p", None)
                dialogue.pop("i_c", None)
                dialogue.pop("i_u", None)

            # Convert conversation to string format with background context
            from utils import conversation_to_str
            try:
                conversation_corpus.append(conversation_to_str(
                    conversation=conversation['dialogues'], 
                    background={
                        'Plot Background': plot['summary'], 
                        'Scenario': conversation['scenario'], 
                        'topic': conversation['topic']
                    }
                ))
            except Exception as e:
                # Log any conversion errors
                from utils import setup_logger
                logger.error(f'Error in conversation_to_str: {e}')
                logger.error(f'Conversation: {conversation}')

    # Define corpus configurations for different retrieval targets
    # Format: (corpus, num_results, corpus_type)
    corpus_map = {
        'raw_text': [(book_corpus, 1, 'book')],
        'expr1': [(experience_corpus, 1, 'experience')],
        'expr3': [(experience_corpus, 3, 'experience')],
        'conv1': [(conversation_corpus, 1, 'conversation')],
        'expr3_conv1': [(experience_corpus, 3, 'experience'), (conversation_corpus, 1, 'conversation')],
        'expr10_conv1': [(experience_corpus, 10, 'experience'), (conversation_corpus, 1, 'conversation')]
    }

    corpora = corpus_map[target]
    retriever = {}

    # Process each corpus configuration
    for (corpus, k, target_type) in corpora:
        # Create document objects
        documents = [Document(page_content=doc) for doc in corpus]

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)

        # Add index metadata to documents
        for i, doc in enumerate(split_docs):
            doc.metadata['idx'] = i

        # Initialize embedding model
        custom_embed_model = Embedding()

        # Create vector store with error handling
        try:
            # Try creating vectorstore all at once
            vectorstore = FAISS.from_documents(split_docs, custom_embed_model)
        except Exception as e:
            print(f"Cannot create vectorstore at once: {e}; will try again.")
            try:
                # Fallback: Create incrementally
                vectorstore = FAISS.from_documents([split_docs[0]], custom_embed_model)
                for doc in split_docs[1:]:
                    vectorstore.add_documents([doc])
                print('Successfully created vectorstore')
            except:
                continue

        # Configure retriever based on k value
        if k != -1:
            # Standard k-nearest neighbor retriever
            retriever[target_type] = vectorstore.as_retriever(search_kwargs={"k": k})
        else:
            # Sequential retriever that returns all documents
            class SequentialRetriever:
                def __init__(self, docs):
                    self.docs = docs
                def invoke(self, query):
                    return self.docs

            # Store original documents and create retriever
            retriever[target_type] = SequentialRetriever(split_docs)

            #retriever[target_type] = vectorstore.as_retriever(search_kwargs={"k": len(split_docs)})
        

    return retriever

def rag(contexts, retriever, target_type):
    """
    Retrieves and formats relevant information from a database based on input contexts.
    
    Args:
        contexts (List[Dict]): List of message dictionaries containing conversation context
        retriever: Document retriever object with invoke() method to fetch relevant docs
        target_type (str): Type of content to retrieve - 'book', 'experience', or 'conversation'
    
    Returns:
        str: Formatted string containing retrieved information with appropriate headers
    """
    # Define headers and titles for different content types
    title_header = {
        'book': '====Book Content====\n\n',
        'experience': '====Historical Experience====\n\n', 
        'conversation': '====Historical Conversation====\n\n'
    }[target_type]
    
    title = {
        'book': 'Content',
        'experience': 'Historical Experience',
        'conversation': 'Historical Conversation'
    }[target_type]

    # Combine context messages into a single query string
    query = "\n\n".join([msg["content"] for msg in contexts])
    
    # Retrieve relevant documents using the retriever
    retrieved_docs = retriever.invoke(query)

    # Sort documents by index if metadata is available
    if retrieved_docs and 'idx' in retrieved_docs[0].metadata:
        retrieved_docs = sorted(retrieved_docs, key=lambda x: x.metadata['idx'])

    # Format the retrieved information
    if len(retrieved_docs) > 1:
        # For multiple documents, include numbered sections
        relevant_info = title_header + ''
        for i, doc in enumerate(retrieved_docs):
            relevant_info +=  f'{title} {i+1}\n' + doc.page_content + '\n\n'
    else:
        # For single document, simple concatenation
        relevant_info = title_header + "\n\n".join([doc.page_content for doc in retrieved_docs])

    return relevant_info

class Agent:
    """
    A conversational agent class that manages dialogue interactions using language models.
    
    This agent can engage in conversations while maintaining context, retrieve relevant information 
    from a knowledge database, and generate appropriate responses based on its configuration.

    Attributes:
        model (str): The language model to use for generating responses
        name (str): Name of the agent/character
        database (Dict): Knowledge database for retrieval
        scene (Dict): Scene context information
        system_prompt (str): Initial system prompt to guide agent behavior
        retrievers (Dict): RAG retrievers for different content types
        system_role (str): Role type for system messages ('user' or 'system')
        messages (List): Conversation history as a list of message dictionaries
    """
    def __init__(self, model: str, name, database: Dict, system_prompt: str = None,
                 scene: Dict = None, retrieval_target: str = 'conversation', 
                 system_prompt_params: Dict = {}, cognitive_system_prompt: str = None):
        # Initialize basic agent properties
        self.model = model 
        self.name = name 
        self.database = database
        self.scene = scene
        self.system_prompt_params = system_prompt_params if system_prompt_params else {}
        self.system_prompt = system_prompt # 可能为 ""
        self.cognitive_system_prompt = cognitive_system_prompt
        # Clean up system prompt by removing trailing newlines
        self.system_prompt = self.system_prompt.strip('\n')

        # Initialize cognitive analysis properties
        self.cognitive_history = []  # 存储历史认知分析结果
        self.last_cognitive_response = None # 上次认知分析的结果
        self.last_cognitive_json = None  # 结构化JSON
        self.last_cognitive_thinking = None  # 思考内容
        self.last_cognitive_formatted = None  # 格式化文本

        self.merge_history = False
        
        # Set up RAG retrievers if database is provided
        if retrieval_target and database:
            self.retrievers = build_rag_corpus(name, database, retrieval_target)
        else:
            self.retrievers = None

        # Add conciseness instruction for non-special characters
        if self.name not in special_characters:
            self.system_prompt = self.system_prompt + '\n\nSpeak concisely as humans, instead of being verbose. Limit your response to 60 words.\n\n'

        # Add name prefix instruction for specific models
        if self.model.startswith('llama') or self.model.startswith('step'):
            self.system_prompt = self.system_prompt + f'Start your response with "{name}: ". Avoid speaking as other characters.\n\n'
        
        # Set appropriate system role based on model type
        if self.model.startswith('claude') or self.model.startswith('o1'):
            self.system_role = 'user'
        else:
            self.system_role = 'system'

        # Initialize conversation history with system prompt
        self.messages = [{"role": self.system_role, "content": self.system_prompt}]


    def chat(self, **kwargs) -> str:
        """
        Generates a response based on the conversation history and available knowledge.
        
        This method:
        1. Retrieves relevant information from the knowledge database if available
        2. Generates a response using the configured language model
        3. Processes and cleans the response based on model-specific requirements
        
        Returns:
            str: The generated response text, or empty string if an error occurs
        """
        response = ""
        try:
            messages = self.messages
            if self.retrievers:
                # Retrieve relevant information from recent context (last 3 messages)
                contexts = self.messages[1:]
                contexts = contexts[-3:]

                # Gather knowledge from all configured retrievers
                knowledge = ''
                for target_type, retriever in self.retrievers.items():
                    knowledge += rag(contexts, retriever, target_type)

                # Insert retrieved knowledge into system prompt
                messages = copy.deepcopy(self.messages)
                messages[0]['content'] = messages[0]['content'].replace('{retrieved_knowledge}', '<begin of background information>\n\n' + knowledge + '\n\n<end of background information>\n\n')

            from utils import get_response_with_retry

            if self.cognitive_system_prompt and self.name not in special_characters:
                cognitive_messages = copy.deepcopy(self.messages)
                cognitive_messages[0]["content"] = self.cognitive_system_prompt
                cognitive_response = get_response_with_retry(
                    model="gpt-4o-nlp", 
                    messages=cognitive_messages, 
                    max_tokens=2048,
                    merge_history=self.merge_history,
                    **kwargs
                )

                cognitive_json, thinking, formatted_cognitive = extract_cognitive_analysis(cognitive_response)

                self.last_cognitive_response = cognitive_response
                self.last_cognitive_json = cognitive_json
                self.last_cognitive_thinking = thinking
                self.last_cognitive_formatted = formatted_cognitive
                self.update_cognitive(cognitive_response)

                if 'system_prompt_params' in dir(self) and self.system_prompt_params:
                    self.system_prompt_params['cognitive_profile'] = formatted_cognitive
                
                # 使用认知分析增强的提示词生成response
                self.system_prompt_params["cognitive_profile"] = formatted_cognitive
                enhanced_prompt = get_character_prompt_w_cognitive_stage2(
                    **self.system_prompt_params.get('function_kwargs', {}),
                    **{k: v for k, v in self.system_prompt_params.items() if k != 'function_kwargs'},
                )

                temp_messages = copy.deepcopy(self.messages)
                temp_messages[0]["content"] = enhanced_prompt + f"{self.system_prompt}"
                response = get_response_with_retry(
                    model=self.model, 
                    messages=temp_messages,
                    max_tokens=1024,
                    merge_history=self.merge_history,
                    **kwargs
                )

            else:
                response = get_response_with_retry(
                    model=self.model, 
                    messages=messages, 
                    max_tokens=2048,
                    merge_history=self.merge_history,
                    **kwargs
                )

            def parse_response(response: str, character_name: str) -> str:
                """
                Args:
                    response (str): Full response text
                    character_name (str): Name of character whose utterance to extract
                
                Returns:
                    str: Extracted utterance for the specified character
                """
                lines = response.split('\n')
                current_character = None
                current_utterance = ""
                parsed_utterances = []

                for line in lines:
                    # Check for character name at start of line
                    if ':' in line:
                        character = line.split(':', 1)[0].strip()
                        
                        if current_character != character:
                            # Save previous character's utterance and start new one
                            if current_utterance:
                                parsed_utterances.append((current_character, current_utterance))
                            current_character = character
                            current_utterance = ""

                    current_utterance += line + "\n"
            
                # Save final utterance
                if current_utterance:
                    parsed_utterances.append((current_character, current_utterance))
                
                parsed_utterances = [utterance for character, utterance in parsed_utterances if character == character_name][0]

                return parsed_utterances

            # Process response based on model type and agent name
            if (self.model.startswith('llama') or self.model.startswith('step')) and self.name != 'NSP':
                response = parse_response(response, self.name)
            
            # Fix repetition issues for certain models
            if not any(self.model.lower().startswith(model_type) for model_type in ['gpt', 'claude']) and self.name != 'NSP':
                res = fix_repeation(response)
                if res:
                    logger.info(f'{self.model} Repetition found and fixed: Original: "{response}" Fixed: "{res}"')
                    response = res

            return response

        except Exception as e:
            import traceback
            print(f"Error getting response: {e}")
            traceback.print_exc()
            
            return ""
    
    def update(self, role: str, message: str):
        """
        Updates the conversation history with a new message.
        
        Args:
            role (str): Role of the message sender
            message (str): Content of the message
        """
        if message:
            # Append message to last message if same role, otherwise add new message
            if self.messages and self.messages[-1]['role'] == role:
                self.messages[-1]['content'] = self.messages[-1]['content'] + '\n\n' + message
            else:
                self.messages.append({"role": role, "content": message})
        return

    def reset(self):
        """
        Resets the conversation history to initial state with only system prompt.
        """
        self.messages = self.messages[:1]


    def update_cognitive(self, cognitive_response: str):
        """
        Updates the agent's cognitive analysis based on the latest response.
        
        Args:
            cognitive_response (str): The latest cognitive analysis result
        """
        self.cognitive_history.append({
            'context': [msg['content'] for msg in self.messages[-3:]] if len(self.messages) > 1 else [],
            'raw_response': cognitive_response,
        })


    def get_last_cognitive_analysis(self):
        """
        Returns the last cognitive analysis result.
        
        Returns:
            str: The last cognitive analysis result
        """
        return {
            'raw_response': self.last_cognitive_response,
            'json_data': self.last_cognitive_json,
            'thinking': self.last_cognitive_thinking,
            'formatted': self.last_cognitive_formatted
        }
    
    def get_cognitive_history(self):
        """
        获取所有历史认知分析结果
        
        Returns:
            list: 包含所有认知分析记录的列表
        """
        return self.cognitive_history
    

    def update_system_prompt(self, new_system_prompt: str):
        """
        动态更新Agent的系统提示。

        这个方法会替换当前的系统提示，并更新消息历史中的第一条消息。
        对于认知更新模式特别有用，可以将前一轮的认知状态整合到新的系统提示中。

        Args:
            new_system_prompt (str): 新的系统提示文本
        """
        # 更新系统提示属性
        self.system_prompt = new_system_prompt.strip('\n')

        # 对于非特殊角色，保持简洁指令
        if self.name not in special_characters:
            self.system_prompt = self.system_prompt + '\n\nSpeak concisely as humans, instead of being verbose. Limit your response to 60 words.\n\n'

        # 为特定模型添加名称前缀指令
        if self.model.startswith('llama') or self.model.startswith('step'):
            self.system_prompt = self.system_prompt + f'Start your response with "{self.name}: ". Avoid speaking as other characters.\n\n'

        # 更新消息历史中的第一条消息
        if self.messages and len(self.messages) > 0:
            self.messages[0] = {"role": self.system_role, "content": self.system_prompt}
        else:
            # 防御性编程：如果没有消息历史，则创建一个新的
            self.messages = [{"role": self.system_role, "content": self.system_prompt}]

        return self.system_prompt
    