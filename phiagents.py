import os
import sys
from dotenv import load_dotenv
from pathlib import Path

from typing import Optional, List, Any
from pydantic import BaseModel, Field

from phi.document import Document
from phi.document.reader.pdf import PDFReader
from phi.assistant import Assistant, AssistantMemory
from phi.knowledge import AssistantKnowledge
from phi.llm.ollama import Ollama
from phi.embedder.ollama import OllamaEmbedder
from phi.vectordb.lancedb import LanceDb
from phi.llm.openai import OpenAIChat
from phi.storage.assistant.sqllite import SqlAssistantStorage
from phi.prompt import PromptTemplate
from phi.llm.groq import Groq
from phi.utils.log import logger

from phi.cli.console import console
from phi.utils.message import get_text_from_message
from phi.utils.timer import Timer

from rich.live import Live
from rich.table import Table
from rich.status import Status
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.box import ROUNDED
from rich.markdown import Markdown
from rich.prompt import Prompt

import pandas as pd
from rich.pretty import pprint
import logging
import json
import re

import lancedb

# from pydantic.tools import parse_obj_as, TypeAdapter

class FollowUpQuestions(BaseModel):
    questions: List[str] = Field(..., description="Lista de preguntas generadas")

    def to_markdown(self):
        return "\n".join([f"- {q}" for q in self.questions])

class LearningObjectives(BaseModel):
    objectives: List[str] = Field(..., description="Lista de objetivos de aprendizaje")

    def to_markdown(self):
        return "\n".join([f"- {o}" for o in self.objectives])

class Contents(BaseModel):
    contents: List[str] = Field(..., description="Lista de contenidos especificos del programa")

    def to_markdown(self):
        return "\n".join([f"- {c}" for c in self.contents])

class MultipleChoiceQuestion(BaseModel):
    question: str = Field(..., description="Pregunta multiple choice generada")
    options: List[str] = Field(..., description="Lista de opciones para la pregunta")
    correct_option: int = Field(..., description="Indice de la opcion correcta")
    hint: str = Field(..., description="Pista para identificar la respuesta correcta")
    justification: str = Field(..., description="Explicacion de por qué la respuesta correcta es la correcta")

    def to_markdown(self):
        return f"{self.question}\n\n" + "\n".join([f"{i+1}. {o}" for i, o in enumerate(self.options)]) 
    # + f"\n\nRespuesta correcta: {self.correct_option}\n\nPista: {self.hint}\n\nJustificación: {self.justification}"

class MultipleChoiceAssessment(BaseModel):
    correct: bool = Field(..., description="Indica si la respuesta a la pregunta es correcta o incorrecta")
    explanation: str = Field(..., description="Explicacion de por qué la respuesta es correcta o incorrecta")
    suggested_topics_to_review: List[str] = Field(..., description="Lista de temas sugeridos para repasar en relacion a la pregunta")

    def to_markdown(self):
        result = '\u2705' if self.correct else '\u274C'
        adicionalmente = ""
        if len(self.suggested_topics_to_review) > 0:
            adicionalmente = "\n\n"+"Adicionalmente, puedes repasar estos temas:\n" + "\n".join([f"- {t}" for t in self.suggested_topics_to_review])
        return f"Resultado: {result}"+"\n\n" + f"{self.explanation}"+adicionalmente

class AgentFactory:

    LLM_MODEL = "gpt-3.5-turbo" # "mixtral-8x7b-32768" # "gpt-3.5-turbo"
    OLLAMA_EMBEDDINGS_MODEL = "nomic-embed-text"

    VECTOR_DATABASE_URI = "./phidata/lancedb"
    BOOK_PDF = "./phidata/andrade_gamboa_corso_la_quimica_esta_entre_nosotros.pdf" 
    OBJECTIVES_PDF = "./phidata/objetivos_de_aprendizaje.pdf"
    CONTENTS_PDF = "./phidata/contenidos.pdf"

    @staticmethod
    def create_qa_agent(name: str = "qa_assistant", 
                    user_id: Optional[str] = None,
                    debug_mode: bool = False, # run_id: Optional[str] = None,
                    knowledge: AssistantKnowledge = None, 
                    read_chat_history: bool = False, 
                    storage_db: Optional[SqlAssistantStorage] = None) -> Assistant:
    
        run_id = None
        if (user_id is not None) and (storage_db is not None):
            existing_run_ids: List[str] = storage_db.get_all_run_ids(user_id)
            if len(existing_run_ids) > 0:
                run_id = existing_run_ids[0]
                read_chat_history = True

        description = "Eres un docente experto en tematicas de Quimica que contesta preguntas para alumnos de una escuela secundaria."
        instructions=[
                "Ante una pregunta del alumno, devuelve una respuesta con informacion sobre dicha pregunta.",
                "Lee cuidadosamente la pregunta y su contexto, y proporciona una respuesta clara y concisa al alumno.",
                "En caso que la pregunta no sea clara, o relevante, o el contexto no sea suficiente, responde 'No lo se, no tengo suficiente contexto para responder la pregunta'.",
                "Para que la pregunta sea relevante, debe ser necesariamente de una temática de Quimica. En caso contrario, responde 'No lo se, no tengo suficiente contexto para responder la pregunta'.",
                "No utilices frases como 'basado en mi conocimiento' o 'dependiendo de la informacion'.",
                "Solo responde sobre la pregunta, NO contestes cosas adicionales. "
            ]
        
        # llm = Groq(model=llm_model)
        # llm = OpenAIChat(model=llm_model), # Ollama(model=llm_model),
        assistant = Assistant(name=name, 
                            run_id=run_id, user_id=user_id,
                            llm=OpenAIChat(model=AgentFactory.LLM_MODEL), 
                            storage=storage_db, knowledge_base=knowledge, 
                            description=description, instructions=instructions, 
                            prevent_hallucinations=True, read_chat_history=read_chat_history,
                            add_references_to_prompt=True, markdown=False,
                            debug_mode=debug_mode, search_knowledge=True)

        return assistant
    
    @staticmethod
    def create_qa_followup_agent(name: str = "qa_followup_assistant", 
                    user_id: Optional[str] = None,
                    debug_mode: bool = False, # run_id: Optional[str] = None,
                    knowledge: AssistantKnowledge = None, 
                    read_chat_history: bool = False, 
                    storage_db: Optional[SqlAssistantStorage] = None) -> Assistant:
    
        run_id = None
        if (user_id is not None) and (storage_db is not None):
            existing_run_ids: List[str] = storage_db.get_all_run_ids(user_id)
            if len(existing_run_ids) > 0:
                run_id = existing_run_ids[0]
                read_chat_history = True

        description = "Eres un docente experto en tematicas de Quimica que genera preguntas para alumnos de una escuela secundaria."
        instructions=[
                "Cuando te provean una pregunta inicial junto con su respuesta, genera 3 preguntas relacionadas a la pregunta inicial.",
                "Lee cuidadosamente la informacion de la pregunta inicial y de su respuesta, y en base a ellas deriva preguntas concisas y DIFERENTES de la pregunta inicial y DIFERENTES entre si.",
                "Solo genera las preguntas relacionadas si la pregunta inicial y su respuesta son relevantes a tematicas de Quimica. En caso contrario, retorna una lista vacia.", 
                "No utilices frases como 'basado en mi conocimiento' o 'dependiendo de la informacion'.",
                "En caso que la pregunta no sea clara, o no este vinculada a la Quimica, o el contexto no sea suficiente, retorna una lista vacia.",
                "Solo genera las 3 preguntas requeridas itemizadas como lista, NO contestes la pregunta inicial y NO agregues comentarios adicionales."
            ]
        
        # llm = Groq(model=llm_model)
        # llm = OpenAIChat(model=llm_model), # Ollama(model=llm_model),

        assistant = Assistant(name=name, 
                            run_id=run_id, user_id=user_id,
                            llm=OpenAIChat(model=AgentFactory.LLM_MODEL), 
                            storage=storage_db, knowledge_base=knowledge, 
                            description=description, instructions=instructions, 
                            prevent_hallucinations=True, read_chat_history=read_chat_history,
                            add_references_to_prompt=True, markdown=False,
                            debug_mode=debug_mode, search_knowledge=True,
                            output_model=FollowUpQuestions)

        return assistant
    
    @staticmethod
    def create_goal_checker_agent(name: str = "objective_checker_assistant", 
                        user_id: Optional[str] = None,
                        debug_mode: bool = False, # run_id: Optional[str] = None,
                        knowledge: AssistantKnowledge = None, 
                        read_chat_history: bool = False,
                        storage_db: Optional[SqlAssistantStorage] = None) -> Assistant:
        
        run_id = None
        if (user_id is not None) and (storage_db is not None):
            existing_run_ids: List[str] = storage_db.get_all_run_ids(user_id)
            if len(existing_run_ids) > 0:
                run_id = existing_run_ids[0]
                read_chat_history = True

        description = "Eres un docente experto en tematicas de Quimica que analiza preguntas y respuestas para alumnos de una escuela secundaria."
        instructions=[
                "Cuando te provean una pregunta junto con su respuesta, identifica como máximo 3 objetivos de aprendizaje del programa que esten mas relacionados a esta pregunta y/o respuesta.",
                "Los objetivos posibles serán recuperados como parte del contexto, y normalmente son mas abstractos que la pregunta y la respuesta." ,
                "Lee cuidadosamente la informacion de la pregunta y de su respuesta para vincularlos con posibles objetivos.",
                "Si consideras que la pregunta y respuesta corresponden a menos objetivos, puedes indicar menos de 3 objetivos.",
                "En caso que la pregunta no sea clara, o no este vinculada a la Quimica, o el contexto no sea suficiente, retorna una lista vacia.",
                "No utilices frases como 'basado en mi conocimiento' o 'dependiendo de la informacion'.",
                "Solo genera los objetivos de aprendizaje como lista, NO contestes la pregunta y NO agregues comentarios adicionales."
            ]
        
        # llm = Groq(model=llm_model)
        # llm = OpenAIChat(model=llm_model), # Ollama(model=llm_model),

        assistant = Assistant(name=name, 
                            run_id=run_id, user_id=user_id,
                            llm=OpenAIChat(model=AgentFactory.LLM_MODEL), 
                            storage=storage_db, knowledge_base=knowledge, 
                            description=description, instructions=instructions, 
                            prevent_hallucinations=True, read_chat_history=read_chat_history,
                            add_references_to_prompt=True, markdown=False,
                            debug_mode=debug_mode, search_knowledge=True,
                            output_model=LearningObjectives)

        return assistant
    
    @staticmethod
    def create_content_checker_agent(name: str = "content_checker_assistant", 
                        user_id: Optional[str] = None,
                        debug_mode: bool = False, # run_id: Optional[str] = None,
                        knowledge: AssistantKnowledge = None, 
                        read_chat_history: bool = False,
                        storage_db: Optional[SqlAssistantStorage] = None) -> Assistant:
        
        run_id = None
        if (user_id is not None) and (storage_db is not None):
            existing_run_ids: List[str] = storage_db.get_all_run_ids(user_id)
            if len(existing_run_ids) > 0:
                run_id = existing_run_ids[0]
                read_chat_history = True

        description = "Eres un docente experto en tematicas de Quimica que analiza preguntas y respuestas para alumnos de una escuela secundaria."
        instructions=[
                "Cuando te provean una pregunta junto con su respuesta, identifica como máximo 3 contenidos del programa que esten mas relacionados a esta pregunta y/o respuesta.",
                "Los contenidos posibles serán recuperados como parte del contexto, y pueden ser mas abstractos que la pregunta y la respuesta." ,
                "Lee cuidadosamente la informacion de la pregunta y de su respuesta para vincularlos con posibles contenidos.",
                "Los contenidos identificados deben ser expresados en forma concisa, idealmente en no mas de 5 palabras.",
                "Si consideras que la pregunta y respuesta corresponden a menos contenidos, puedes indicar menos de 3 contenidos.",
                "En caso que la pregunta no sea clara o el contexto no sea suficiente, retorna una lista vacia.",
                "No utilices frases como 'basado en mi conocimiento' o 'dependiendo de la informacion'.",
                "Solo genera los contenidos identificados como lista, NO contestes la pregunta y NO agregues comentarios adicionales."
            ]
        
        # llm = Groq(model=llm_model)
        # llm = OpenAIChat(model=llm_model), # Ollama(model=llm_model),

        assistant = Assistant(name=name, 
                            run_id=run_id, user_id=user_id,
                            llm=OpenAIChat(model=AgentFactory.LLM_MODEL), 
                            storage=storage_db, knowledge_base=knowledge, 
                            description=description, instructions=instructions, 
                            prevent_hallucinations=True, read_chat_history=read_chat_history,
                            add_references_to_prompt=True, markdown=False,
                            debug_mode=debug_mode, search_knowledge=True,
                            output_model=Contents)

        return assistant
    
    @staticmethod
    def create_multiplechoice_agent(name: str = "multiple_choice_assistant", user_id: Optional[str] = None,
                      debug_mode: bool = False, # run_id: Optional[str] = None,
                      knowledge: AssistantKnowledge = None, 
                      read_chat_history: bool = False,
                      storage_db: Optional[SqlAssistantStorage] = None) -> Assistant:
    
        run_id = None
        if (user_id is not None) and (storage_db is not None):
            existing_run_ids: List[str] = storage_db.get_all_run_ids(user_id)
            if len(existing_run_ids) > 0:
                run_id = existing_run_ids[0]
                read_chat_history = True

        description = "Eres un docente experto en tematicas de Quimica que genera preguntas de opcion multiple para alumnos de una escuela secundaria."
        instructions=[
                "Dado un tema particular sobre Quimica, analizalo en detalle para determinar su relevancia y relación con el contexto provisto.",
                "En caso que determines que el tema NO ES relevante a Quimica o el contexto provisto no es suficiente, NO generes pregunta ni opciones, y responde 'No tengo suficiente contexto para generar una pregunta multiple-choice.' como pregunta, con justificacion vacia, pista vacia, provee una lista vacia de opciones, y -1 como opcion a escoger.",
                "En caso que determines que el tema efectivamente ES relevante a Quimica, genera 1 pregunta de tipo opcion multiple (multiple choice) que permita evaluar el conocimiento del alumno sobre el tema.",
                "Para esta pregunta, genera además exactamente 4 opciones de respuesta, donde solo una de dichas opciones contesta correctamente la pregunta.",
                "Indica cuál de las 4 opciones es la correcta.",
                "Provee una pista que permita al alumno identificar la respuesta correcta.",
                "Da una explicación breve de por qué la respuesta identificada como correcta es la correcta.",
                "Solo genera la pregunta y sus opciones de respuesta si determinaste que el tema es de relevancia para la Quimica. "
                "Cuando generes la pregunta requerida y sus opciones en el formato establecido, NO agregues comentarios adicionales."
            ]
        
        # llm = Groq(model=llm_model)
        # llm = OpenAIChat(model=llm_model), # Ollama(model=llm_model),

        assistant = Assistant(name=name, 
                            run_id=run_id, user_id=user_id,
                            llm=OpenAIChat(model=AgentFactory.LLM_MODEL), 
                            storage=storage_db, knowledge_base=knowledge, 
                            description=description, instructions=instructions, 
                            prevent_hallucinations=True, read_chat_history=read_chat_history,
                            add_references_to_prompt=True, markdown=False,
                            debug_mode=debug_mode, search_knowledge=True,
                            output_model=MultipleChoiceQuestion)

        return assistant
    
    def create_multiplechoice_corrector_agent(name: str = "multiple_choice_corrector_assistant", 
                        user_id: Optional[str] = None,
                        debug_mode: bool = False, # run_id: Optional[str] = None,
                        knowledge: AssistantKnowledge = None, 
                        read_chat_history: bool = False,
                        storage_db: Optional[SqlAssistantStorage] = None) -> Assistant:
    
        run_id = None
        if (user_id is not None) and (storage_db is not None):
            existing_run_ids: List[str] = storage_db.get_all_run_ids(user_id)
            if len(existing_run_ids) > 0:
                run_id = existing_run_ids[0]
                read_chat_history = True

        description = "Eres un docente experto en tematicas de Quimica que corrige preguntas de opcion multiple para alumnos de una escuela secundaria."
        instructions=[
                "Dada una pregunta y una posible respuesta para la pregunta dada por el alumno, evalua si esta respuesta es correcta o no.",
                "Como ayuda, recibirás tambien la respuesta correcta para la pregunta multiple-choice."
                "Para tu evaluacion, lee cuidadosamente la informacion de la pregunta y de ambas respuestas en relación al contexto provisto.",
                "Da una explicación breve de por qué la respuesta del alumno es correcta o incorrecta en base al contexto provisto.",
                "Si la respuesta del alumno es incorrecta, explica brevemente cuál es la respuesta correcta para la pregunta."
                "En caso que la pregunta no sea clara o el contexto no sea suficiente, responde 'No tengo suficiente contexto para evaluar la pregunta multiple-choice.'.",
                "Adicionalmente, sugiere hasta 3 temas que el alumno puede repasar para contestar la temática de la pregunta."
            ]
        
        # llm = Groq(model=llm_model)
        # llm = OpenAIChat(model=llm_model), # Ollama(model=llm_model),

        assistant = Assistant(name=name, 
                            run_id=run_id, user_id=user_id,
                            llm=OpenAIChat(model=AgentFactory.LLM_MODEL), 
                            storage=storage_db, knowledge_base=knowledge, 
                            description=description, instructions=instructions, 
                            prevent_hallucinations=True, read_chat_history=read_chat_history,
                            add_references_to_prompt=True, markdown=False,
                            debug_mode=debug_mode, search_knowledge=True,
                            output_model=MultipleChoiceAssessment)

        return assistant
    
    @staticmethod
    def create_knowledge_base(uri: str, table_name: str, top_k: int = 3, pdf_filename: str = None, 
                              upsert: bool = False, embeddings=None) -> AssistantKnowledge:
        
        # Note: it requires Ollama server running in background!
        if embeddings is None:
            embeddings = OllamaEmbedder(model=AgentFactory.OLLAMA_EMBEDDINGS_MODEL, dimensions=768)

        db = lancedb.connect(uri)
        connection = None
        # print(db.table_names())
        if table_name in db.table_names():
            connection = db.open_table(table_name)
            vector_database = LanceDb(uri=uri, embedder=embeddings, connection=connection)
        else:
            vector_database = LanceDb(uri=uri, embedder=embeddings, table_name=table_name)
        
        # vector_database = LanceDb(uri=uri, embedder=embeddings, table_name=table_name)
        # print(vector_database._id) "id"
        
        # Define the knowledge base
        knowledge = AssistantKnowledge(vector_db=vector_database,
            # 3 references are added to the prompt
            num_documents=top_k,
        )
       
        if pdf_filename:
            reader = PDFReader()
            rag_documents: List[Document] = reader.read(pdf_filename)
            if rag_documents:
                print(f"Loading {pdf_filename} into {table_name} database (vectors:", len(rag_documents),") ...")
                knowledge.load_documents(rag_documents, upsert=upsert)
        
        return knowledge

    @staticmethod
    def load_knowledge_base(knowledge: AssistantKnowledge, pdf_filename: str, upsert: bool = False) -> AssistantKnowledge:
        
        reader = PDFReader()
        rag_documents: List[Document] = reader.read(pdf_filename)
        if rag_documents:
            print(f"Loading {pdf_filename} into database (vectors:", len(rag_documents),") ...")
            knowledge.load_documents(rag_documents, upsert=True)
  
        return knowledge


class BaseAssistant:

    def load_databases(self, logging: bool = False):
        pass

    def query(self, question: str) -> str:
        pass

    def run_loop(self, exit_on: Optional[List[str]] = ["exit", "quit", "salir"], 
                 user: str = "Pregunta", emoji: str = ":sunglasses:",
                 welcome_message: Optional[str] = None):

        if welcome_message:
            console.print(welcome_message)

        while True:
            message = Prompt.ask(f"[bold]{emoji} {user}[/bold]")
            if message in exit_on:
                console.print("")
                break
            response = self.query(message)
            # pprint(response)
            console.print("")
        
        console.print(":thumbs_up: Bye! \n")

    @staticmethod
    def convert_string_to_json(s):
        s = s.replace('\n','')
        pattern = r'\{(.*?)\}'
        match = re.search(pattern, s)
        if match:
            s = match.group(1)
            return json.loads("{"+s+"}")
        return None
    
    @staticmethod
    def convert_response_to_string(self, response: Any) -> str:
        if isinstance(response, str):
            return response
        elif isinstance(response, BaseModel):
            return response.model_dump_json(exclude_none=True, indent=4)
        else:
            return json.dumps(response, indent=4)
    
    @staticmethod
    def print_response(response, markdown=True, message=None, response_timer=None):
        
        _response = Markdown(response) if markdown else BaseAssistant.convert_response_to_string(response)

        table = Table(box=ROUNDED, border_style="blue", show_header=False)
        if message:
            table.show_header = True
            table.add_column("Message")
            table.add_column(get_text_from_message(message))
        if response_timer:
            table.add_row(f"Response\n({response_timer.elapsed:.1f}s)", _response)  # type: ignore
        else:
            table.add_row(f"Response\n()", _response)  # type: ignore
        console.print(table)

    @staticmethod
    def execute_query(agent, query, coerce=True):
        response_timer = Timer()
        response_timer.start()
        response = None
        with Progress(
            SpinnerColumn(spinner_name="dots"), TextColumn("{task.description}"), transient=True
        ) as progress:
            progress.add_task("Working...")
            response = agent.run(query, stream=False)
            if isinstance(response, str) and coerce:
                # print("Format error:", response)
                response = BaseAssistant.convert_string_to_json(response)
        
        response_timer.stop()
        return response, response_timer
