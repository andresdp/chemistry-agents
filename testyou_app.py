import os
from dotenv import load_dotenv
from pathlib import Path

from typing import Optional, List, Any
from pydantic import BaseModel, Field

from phi.utils.log import logger

from rich.prompt import Prompt
from rich.pretty import pprint
from phi.cli.console import console

from phiagents import BaseAssistant, AgentFactory
from phiagents import MultipleChoiceQuestion, MultipleChoiceAssessment
from phiagents import LearningObjectives, Contents

class DTOResponse(BaseModel):
    topic: str = Field(..., description="Tema para la pregunta")
    question: str = Field(..., description="Pregunta multiple choice generada")
    options: List[str] = Field(..., description="Lista de opciones para la pregunta")
    correct_option: int = Field(..., description="Indice de la opcion correcta")
    hint: str = Field(..., description="Pista para identificar la respuesta correcta")
    answer: str = Field(..., description="Respuesta dada por el usuario")
    correct: bool = Field(..., description="Indica si la respuesta a la pregunta es correcta o incorrecta")
    explanation: str = Field(..., description="Explicacion de por qué la respuesta es correcta o incorrecta")
    suggested_topics_to_review: List[str] = Field(..., description="Lista de temas sugeridos para repasar en relacion a la pregunta")
    objectives: List[str] = Field(..., description="Lista de objetivos de aprendizaje relacionados")
    contents: List[str] = Field(..., description="Lista de contenidos relacionados")

class TestYouAssistant(BaseAssistant):

    MEMORY_FILE = ".//phidata//storage.db"
    WELCOME_MESSAGE = "Soy QuimiBot, tu asistente de Quimica! :robot: Voy a generar preguntas para evaluar tus conocimientos ...\n"

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
        self.multiple_choice_agent = AgentFactory.create_multiplechoice_agent(knowledge=self.book_database, storage_db=self.memory_database, read_chat_history=True) 
        self.multiple_choice_corrector_agent = AgentFactory.create_multiplechoice_corrector_agent(knowledge=self.book_database, storage_db=self.memory_database, read_chat_history=True) 
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

        # TODO: Revisar el flujo de este workflow, cuando no se genera una pregunta multiple choice
        # y como se evalua/muestra la respuesta correcta

        dto_response = dict()
        dto_response['topic'] = question

        # 1. Generate a multiple choice question for the topic
        new_question = "Tema: "+question
        response, timer = self.execute_query(self.multiple_choice_agent, new_question)
        if isinstance(response, dict):
            response = MultipleChoiceQuestion(**response)     
        # pprint(response)
        self.print_response(response.to_markdown(), response_timer=timer, message=new_question)
        dto_response['question'] = response.question
        dto_response['options'] = response.options
        dto_response['correct_option'] = response.correct_option
        dto_response['hint'] = response.hint

        dto_response['answer'] = ""
        dto_response['correct'] = False
        dto_response['explanation'] = ""
        dto_response['suggested_topics_to_review'] = []
        dto_response['objectives'] = []
        dto_response['contents'] = []
        # Warning: the question could come up empty, so we need to check for that
        if response.correct_option > -1:
            # 2. Ask the user for an answer (option) for the question
            message = Prompt.ask(f"[bold]Tu respuesta[/bold]")
            answer = "Invalida"
            if message.isdigit():
                n_user = int(message)
                if (n_user > 0) and (n_user <= 4) and len(response.options) >= n_user:
                    answer = response.options[n_user-1]
            dto_response['answer'] = answer
            m_correct = response.correct_option
            correct_answer = response.options[m_correct-1]

            # 3. Check if the answer is correct
            new_question = "Pregunta: "+question+" \nRespuesta del estudiante: "+answer + "\nRespuesta correcta: "+correct_answer
            # print(new_question)
            console.print("\nRespuesta correcta: "+correct_answer)
            response, timer = self.execute_query(self.multiple_choice_corrector_agent, new_question)
            if isinstance(response, dict):
                response = MultipleChoiceAssessment(**response)
            # pprint(response)
            self.print_response(response.to_markdown(), response_timer=timer, message=answer)
            if message.isdigit():
                dto_response['correct'] = (n_user == m_correct) # response.correct
            else:
                dto_response['correct'] = False
            dto_response['explanation'] = response.explanation
            dto_response['suggested_topics_to_review'] = response.suggested_topics_to_review

            new_question = "Pregunta: "+question+" \nRespuesta: "+correct_answer
            # 4. Identify learning objectives for the question/answer
            response, timer = self.execute_query(self.goal_checker_agent, new_question)
            if isinstance(response, dict):
                response = LearningObjectives(**response)
            # pprint(response)
            message = "Objetivos de aprendizaje relacionados"
            self.print_response(response.to_markdown(), response_timer=timer, message=message)
            dto_response['objectives'] = response.objectives

            # 5. Identify contents for the question/answer
            response, timer = self.execute_query(self.content_checker_agent, new_question)
            if isinstance(response, dict):
                response = Contents(**response)
            # pprint(response)
            message = "Contenidos relacionados"
            self.print_response(response.to_markdown(), response_timer=timer, message=message)
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

    assistant = TestYouAssistant()
    # assistant.load_databases(logging=False) # This is only needed to create the databases the first time
    print()

    assistant.run_loop(welcome_message=TestYouAssistant.WELCOME_MESSAGE, user="Tema")

    print()