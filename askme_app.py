import os
from dotenv import load_dotenv
from pathlib import Path

from typing import Optional, List, Any
from pydantic import BaseModel, Field

from phi.utils.log import logger

from phiagents import BaseAssistant, AgentFactory
from phiagents import FollowUpQuestions, LearningObjectives, Contents

from rich.pretty import pprint

class DTOResponse(BaseModel):
    question: str = Field(..., description="Pregunta realizada")
    answer: str = Field(..., description="Respuesta a la pregunta")
    follow_up_questions: List[str] = Field(..., description="Lista de preguntas generadas")
    objectives: List[str] = Field(..., description="Lista de objetivos de aprendizaje")
    contents: List[str] = Field(..., description="Lista de contenidos especificos del programa")

class AskMeAssistant(BaseAssistant):

    MEMORY_FILE = ".//phidata//storage.db"
    WELCOME_MESSAGE = "Soy QuimiBot, tu asistente de Quimica! :robot: Preguntame cualquier cosa ...\n"

    def __init__(self, logging: bool = False):

        logger.disabled = not logging

        # For storing chat history
        # storage_db = SqlAssistantStorage(table_name="rag_assistant", db_file=MEMORY_FILE)
        # storage_db.create()
        self.memory_database = None

        # Create the knowledge bases
        self.book_database = AgentFactory.create_knowledge_base(uri=AgentFactory.VECTOR_DATABASE_URI, table_name="book")
        self.objectives_database = AgentFactory.create_knowledge_base(uri=AgentFactory.VECTOR_DATABASE_URI, table_name="objectives")
        self.contents_database = AgentFactory.create_knowledge_base(uri=AgentFactory.VECTOR_DATABASE_URI, table_name="contents")

        # Create the agents
        self.qa_agent = AgentFactory.create_qa_agent(knowledge=self.book_database, storage_db=self.memory_database, read_chat_history=True) 
        self.qa_followup_agent = AgentFactory.create_qa_followup_agent(knowledge=self.book_database, storage_db=self.memory_database, read_chat_history=True) 
        self.goal_checker_agent = AgentFactory.create_goal_checker_agent(knowledge=self.objectives_database, storage_db=self.memory_database, read_chat_history=True) 
        self.content_checker_agent = AgentFactory.create_content_checker_agent(knowledge=self.contents_database, storage_db=self.memory_database, read_chat_history=True) 

    def load_databases(self, logging: bool = False):

        logger.disabled = not logging

        # Load the PDF files into the databases for each agent
        self.book_database = AgentFactory.load_knowledge_base(self.book_database, AgentFactory.BOOK_PDF)
        self.objectives_database = AgentFactory.load_knowledge_base(self.objectives_database, AgentFactory.OBJECTIVES_PDF)
        self.contents_database = AgentFactory.load_knowledge_base(self.contents_database, AgentFactory.CONTENTS_PDF)

    def query(self, question: str) -> str:
        # Main agentic workflow for the assistant

        # 1. Ask the question to the agent
        answer, timer = self.execute_query(self.qa_agent, question, coerce=False)
        self.print_response(answer, response_timer=timer, message=question)
        # pprint(answer)

        dto_response = dict()
        dto_response['question'] = question
        dto_response['answer'] = answer

        # 2. Derove follow-up questions
        new_question = "Pregunta: "+question+" \nRespuesta: "+answer
        response, timer = self.execute_query(self.qa_followup_agent, new_question)
        if isinstance(response, dict):
            response = FollowUpQuestions(**response)
        message = "Preguntas relacionadas"
        self.print_response(response.to_markdown(), response_timer=timer, message=message)
        # pprint(response)
        dto_response['follow_up_questions'] = response.questions

        # 3. Identify learning objectives for the question/answer
        response, timer = self.execute_query(self.goal_checker_agent, new_question)
        if isinstance(response, dict):
            response = LearningObjectives(**response)
        message = "Objetivos de aprendizaje relacionados"
        self.print_response(response.to_markdown(), response_timer=timer, message=message)
        # pprint(response)
        dto_response['objectives'] = response.objectives

        # 4. Identify contents for the question/answer
        response, timer = self.execute_query(self.content_checker_agent, new_question)
        if isinstance(response, dict):
            response = Contents(**response)
        message = "Contenidos relacionados"
        self.print_response(response.to_markdown(), response_timer=timer, message=message)
        # pprint(response)
        dto_response['contents'] = response.contents

        # pprint(DTOResponse(**dto_response))
        return DTOResponse(**dto_response)


# --- MAIN ---

if __name__ == "__main__":

    # Configuring OpenAI (GPT)
    print()
    load_dotenv()
    # CONFIGURE YOUR OPENAI_API_KEY HERE!
    os.environ["OPENAI_MODEL_NAME"] = os.getenv('LLM_MODEL') # 'gpt-3.5-turbo'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    assistant = AskMeAssistant()
    # assistant.load_databases(logging=False) # This is only needed to create the databases the first time
    print()

    assistant.run_loop(welcome_message=AskMeAssistant.WELCOME_MESSAGE)

    print()