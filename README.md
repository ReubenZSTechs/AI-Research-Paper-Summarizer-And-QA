<a id="top"></a>

# About the project
*This markdown file acts as a note to the architecture of the model, the pattern used, the type of models used, future works to be done, and more.*

<details>
<summary><strong>Table of Contents</strong></summary>

- [Background](#background)
- [Key Features](#key-features)
- [Langgraph Architecture](#langgraph-architecture)
- [LLM Models and Roles](#llm-models-and-roles)
- [Node Tasks](#node-tasks)
- [Tools](#tools)
- [Metrics Used](#metrics-used)
- [Future Work](#future-work)

</details>

---

# Background
Reading a journal paper especially those with huge amount of pages can be time-wasting. On top of that, summarizing or taking notes from a journal paper means exhausting double the effort: reading and summarizing. For that, this project will act as a research summarizer agent that summarizes keypoints from a given journal as well as injecting knowledge from the internet to further enrich the quality of the summary. Do note that this summary does not cover the entire scope of the journal so still further reading is required.

<div align="right">
  <a href="#top">Back to top ⬆</a>
</div>

---

# Key Features
- **Custom-built Agent Architecture**<br>Featuring langgraph architecture for agent and state management across the entire process
- **Integration with search API**<br>Integrating DuckDuckGo search library for unlimited search queries
- **Local LLM Hosting**<br>Leveraging Ollama to host local LLMs, such as Mistral, Gemma, and Llama for reasoning and working models
- **Multi LLM Model Architecture**<br>Utilizes multi LLM local models for specific tasks assigned later.


<div align="right">
  <a href="#top">Back to top ⬆</a>
</div>

---

# Langgraph Architecture
The proposed Langgraph Architecture for initial phase is as follows

```text
PDF PATH INPUT --> INGESTION --> UNDERSTANDING --> REASONING --> FORMATTING --> EVALUATION --> PDF OUTPUT
```
Ingestion Phase includes:
```text
LOAD PDF --> PARSE SECTIONS --> CHUNKING
```

Understanding Phase includes:
```text
SUMMARIZING CHUNKS --> AGGREGATE SECTIONS
```

Reasoning Phase includes:
```text
DETECT RESEARCH GAPS --> INTERNET RETRIEVAL --> SYNTHESIZING --> SELF-CRITIC
```

Formatting Phase includes:
```text
FORMAT NOTES
```

Evaluation Phase includes:
```text
EVALUATE OUTPUT
```

Note: Should the metric faithfulness scores the output to be less than the threshold of 80%, then the model will restart again from the reasoning phase, specifically the synthesizing, of course with different prompt.

<div align="right">
  <a href="#top">Back to top ⬆</a>
</div>

---

# Node Tasks
Each node, starting from `Understanding Phase` up to the `Evaluation Phase` will be limited to the scope of the defined tasks, but not limit to as follows.

<table>
    <tr>
        <td><b>Node</b></td>
        <td><b>Model</b></td>
        <td><b>Reason</b></td>
    </tr>
    <tr>
        <td>Summarize Chunks</td>
        <td>Worker</td>
        <td>Bulk Processing Related task</td>
    </tr>
    <tr>
        <td>Aggregate Sections</td>
        <td>Worker</td>
        <td>Structured Merging</td>
    </tr>
    <tr>
        <td>Gap Detector</td>
        <td>Reasoner</td>
        <td>Requires reasoning to detect gaps</td>
    </tr>
    <tr>
        <td>Internet Retrieval</td>
        <td>Worker</td>
        <td>Simple retrieval and compression</td>
    </tr>
    <tr>
        <td>Synthesize</td>
        <td>Reasoner</td>
        <td>Requires reasoning to synthesze internet data and sections</td>
    </tr>
    <tr>
        <td>Self-critic</td>
        <td>Reasoner</td>
        <td>Critique + Internet re-retrieval and resynthesize for improvement</td>
    </tr>
        <tr>
        <td>Format Nodes</td>
        <td>Worker</td>
        <td>Formatting notes</td>
    </tr>
    <tr>
        <td>Evaluate Output</td>
        <td>Evaluator</td>
        <td>Unbiased Scorer</td>
    </tr>
</table>


<div align="right">
  <a href="#top">Back to top ⬆</a>
</div>

---

# LLM Models and Roles
In the initial phase, the proposed roles and LLM models assigned are as follows.

<table>
    <tr>
        <td><b>Role</b></td>
        <td><b>Model</b></td>
        <td><b>Parameter Count</b></td>
    </tr>
    <tr>
        <td>Worker</td>
        <td>Mistral-Nemo</td>
        <td>12B parameters</td>
    </tr>
    <tr>
        <td>Reasoner</td>
        <td>qwen3</td>
        <td>14B parameters</td>
    </tr>
    <tr>
        <td>Evaluator</td>
        <td>Llama3.2</td>
        <td>3B parameters</td>
    </tr>
</table>


<div align="right">
  <a href="#top">Back to top ⬆</a>
</div>

---

# Tools
- In the `Internet Retrieval` node, a web searching tool, specifically DDGS (DuckDuckGo Search) library will be equipped to search the internet regarding the query in question.


<div align="right">
  <a href="#top">Back to top ⬆</a>
</div>

---

# Metrics Used
To evaluate the performance of the model, the `Evaluate Output` node and other functions will be defined to cover the following metrics.

<table>
    <tr>
        <td><b>Metric</b></td>
        <td><b>What does it measure?</b></td>
    </tr>
    <tr>
        <td>Latency</td>
        <td>Measure the starting time and end time of execution per node</td>
    </tr>
    <tr>
        <td>Web Calls</td>
        <td>Determine how many times the DDGS library is called</td>
    </tr>
    <tr>
        <td>Faithfulness</td>
        <td>How consistent is the summary to the actual facts</td>
    </tr>
        <tr>
        <td>Relevance</td>
        <td>Is the overall summary relevant to the key points</td>
    </tr>
    <tr>
        <td>Completeness</td>
        <td>How complete is the summary to the key points extracted</td>
    </tr>
    <tr>
        <td>Total Runtime</td>
        <td>Calculates the time it takes to reach from the starting point to the endpoint</td>
    </tr>
</table>


<div align="right">
  <a href="#top">Back to top ⬆</a>
</div>

---

# Future Work

- [ ] Implement a debater pattern in the `Synthesize` node and/ or the `Self-critic` node
- [ ] Implement a planning agent using the supervisor pattern at the beginning
- [ ] Implement a vector knowledge database to store all reports and generate summary report when needed
- [ ] Upgrade the system to be a chatbot thus implementing a RAG pipeline module

<div align="right">
  <a href="#top">Back to top ⬆</a>
</div>

---