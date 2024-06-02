# Chemistry Agents
A demo about an agentic workflow applied to a chemistry classroom.

This proof-of-concept implements 2 assistants:

* **AskMe**: It provides support for Q&A from students about chemistry topics. The questions are anwsered from a chemistry textbook.
* **TestYou**: It can generate multiple-choice questions for an input topic. Once the student answers the question, it is automatically graded.

<!---
![](assistants.png)
-->

<p align="center">
<img src="assistants.png" width="700"/>
</p>

In addition, the assistants can link the questions and answers to predefined learning goals and course contents.

Both assistants are internally designed as agentic workflows using the [Phidata](https://www.phidata.com/) framework.

### AskMe Assistant
To run it in command line:
```python 
python askme_app.py
```
and you should see the following interactions:

```console
Soy QuimiBot, tu asistente de Quimica! 🤖 Preguntame cualquier cosa ...

😎 Pregunta: Cual es la diferencia entre atomos y moleculas?
╭──────────┬────────────────────────────────────────────────────────────────────────────────────╮
│ Message  │ Cual es la diferencia entre atomos y moleculas?                                    │
├──────────┼────────────────────────────────────────────────────────────────────────────────────┤
│ Response │ Los átomos son las unidades más pequeñas de un elemento químico y no se dividen en │
│ (3.2s)   │ partes más pequeñas sin perder sus propiedades. Las moléculas, por otro lado,      │
│          │ consisten en la unión de dos o más átomos diferentes, y pueden ser la unidad más   │
│          │ pequeña de un compuesto químico que conserva las propiedades de ese compuesto.     │
│          │ Además, la forma de los átomos es generalmente esférica, mientras que las formas   │
│          │ moleculares son el resultado de la aglomeración espacial de los átomos             │
│          │ constituyentes.                                                                    │
╰──────────┴────────────────────────────────────────────────────────────────────────────────────╯
╭──────────┬────────────────────────────────────────────────────────────────────────────────────╮
│ Message  │ Preguntas relacionadas                                                             │
├──────────┼────────────────────────────────────────────────────────────────────────────────────┤
│ Response │                                                                                    │
│ (3.1s)   │  • ¿Cuál es la importancia de la forma esférica de los átomos?                     │
│          │  • ¿Cómo se mantienen unidos los átomos en una molécula?                           │
│          │  • ¿Por qué la unión de los átomos en una molécula conserva las propiedades del    │
│          │    compuesto químico?                                                              │
╰──────────┴────────────────────────────────────────────────────────────────────────────────────╯
╭──────────┬────────────────────────────────────────────────────────────────────────────────────╮
│ Message  │ Objetivos de aprendizaje relacionados                                              │
├──────────┼────────────────────────────────────────────────────────────────────────────────────┤
│ Response │                                                                                    │
│ (2.5s)   │  • Identificar el conjunto de variables relevantes para la explicación del         │
│          │    comportamiento de diversos sistemas químicos                                    │
│          │  • Utilizar conceptos, modelos y procedimientos de la Química en la resolución de  │
│          │    problemas cualitativos y cuantitativos relacionados con los ejes temáticos      │
│          │    trabajados                                                                      │
│          │  • Establecer relaciones de pertinencia entre los datos experimentales relevantes  │
│          │    y los modelos teóricos correspondientes                                         │
╰──────────┴────────────────────────────────────────────────────────────────────────────────────╯
╭──────────┬────────────────────────────────────────────────────────────────────────────────────╮
│ Message  │ Contenidos relacionados                                                            │
├──────────┼────────────────────────────────────────────────────────────────────────────────────┤
│ Response │                                                                                    │
│ (1.7s)   │  • Estructura Atómica                                                              │
│          │  • Moléculas                                                                       │
│          │  • Uniones Químicas                                                                │
╰──────────┴────────────────────────────────────────────────────────────────────────────────────╯
```

### TestYou Assistant
To run it in command line:
```python 
python testyou_app.py
```
and you should see the following interactions:

```console
Soy QuimiBot, tu asistente de Quimica! 🤖 Voy a generar preguntas para evaluar tus conocimientos ...

😎 Tema: Aleaciones metalicas
╭──────────┬───────────────────────────────────────────────────────────────────────────────────────────╮
│ Message  │ Tema: Aleaciones metalicas                                                                │
├──────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ Response │ ¿Qué tipo de interacción se da exclusivamente en moléculas que poseen átomos de hidrógeno │
│ (9.3s)   │ unidos a átomos muy negativos, como el oxígeno y el nitrógeno?                            │
│          │                                                                                           │
│          │  1 Enlaces covalentes                                                                     │
│          │  2 Puentes de hidrógeno                                                                   │
│          │  3 Enlaces iónicos                                                                        │
│          │  4 Fuerzas de London                                                                      │
│          │  5 Enlaces metálicos                                                                      │
╰──────────┴───────────────────────────────────────────────────────────────────────────────────────────╯
Tu respuesta: 2

Respuesta correcta: Enlaces covalentes
╭──────────┬───────────────────────────────────────────────────────────────────────────────────────────╮
│ Message  │ Puentes de hidrógeno                                                                      │
├──────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ Response │ Resultado: ❌                                                                             │
│ (4.4s)   │                                                                                           │
│          │ La respuesta del estudiante 'Puentes de hidrógeno' es incorrecta porque las aleaciones    │
│          │ metálicas se forman a partir de la mezcla de dos o más metales, no de puentes de          │
│          │ hidrógeno. Las aleaciones metálicas son una solución sólida de un metal en otro metal o   │
│          │ en una mezcla de metales.                                                                 │
│          │                                                                                           │
│          │ Adicionalmente, puedes repasar estos temas:                                               │
│          │                                                                                           │
│          │  • Aleaciones metálicas                                                                   │
│          │  • Enlaces covalentes                                                                     │
│          │  • Química de los metales                                                                 │
╰──────────┴───────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────┬───────────────────────────────────────────────────────────────────────────────────────────╮
│ Message  │ Objetivos de aprendizaje relacionados                                                     │
├──────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ Response │                                                                                           │
│ (1.7s)   │  • Identificar el conjunto de variables relevantes para la explicación del comportamiento │
│          │    de diversos sistemas químicos                                                          │
│          │  • Establecer relaciones de pertinencia entre los datos experimentales relevados y los    │
│          │    modelos teóricos correspondientes                                                      │
╰──────────┴───────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────┬───────────────────────────────────────────────────────────────────────────────────────────╮
│ Message  │ Contenidos relacionados                                                                   │
├──────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ Response │                                                                                           │
│ (1.1s)   │                                                                                           │
╰──────────┴───────────────────────────────────────────────────────────────────────────────────────────╯
```
