# DecisionArticulationLLM

DecisionArticulationLLM is a Python-based research application for studying whether people can effectively articulate their decision context to an LLM, and whether iterative clarification questions help improve the quality of structured decision analysis outputs.

The project supports two ways to use the system:

- **Local CLI version**: run the main Python entry point locally.
- **Streamlit web app**: use the deployed online interface.

Online app:

https://decisionarticulationllm-kfpfrparpycckbaymdv3yy.streamlit.app/

## Research Goal

This project investigates the following question:

> Can users effectively express their decision situation to an AI system, and does an iterative clarification process help the AI produce a more faithful, complete, clear, and useful decision representation?

The application is designed for decision analysis research. Instead of only producing a final recommendation, it studies how a user's decision narrative is transformed into structured decision information across multiple interaction rounds.

## Core Workflow

1. **User enters a decision**
   - Decision title
   - Decision narrative describing the situation, goals, constraints, concerns, stakeholders, and uncertainties

2. **Initial LLM generation**
   - The system generates an initial structured decision representation.

3. **User evaluates the current round**
   - The user rates the output on five criteria:
     - Faithfulness to the real situation
     - Completeness
     - Clarity
     - Usefulness for decision-making
     - Helpfulness for expressing what the user meant

4. **Clarification question generation**
   - The system asks targeted follow-up questions based on the current output.

5. **User answers clarification questions**
   - The user provides additional information.

6. **Refinement**
   - The system updates the structured decision output using the user's answers.

7. **AI improvement evaluation**
   - After each refinement round, the app compares the new output with the previous round and evaluates whether the clarification process improved or worsened the result.

8. **Export**
   - The system saves or exports study results for later analysis.

## Structured Output

For each round, the system produces a structured decision representation containing:

- **Decision Summary**
- **Alternatives**
- **Preferences**
- **Uncertainties**
- **Ethical Issues**
- **Stakeholders**
- **Missing but Relevant Information**
- **Refinement Notes**

Each item can also include change tracking, such as whether it was unchanged, revised, or added in a later round.

## Evaluation Design

### User Evaluation

Users rate each round on a 1-5 scale across the following dimensions:

| Dimension | Description |
|---|---|
| Faithfulness | How well the output matches the user's real situation |
| Completeness | How much relevant decision information is captured |
| Clarity | How understandable and well-organized the output is |
| Usefulness | How helpful the output is for decision-making |
| Self-expression Support | Whether the system helped the user express what they meant |

### AI Improvement Evaluation

Starting from the first refinement round, the system also generates an AI-based improvement evaluation comparing the current round with the previous round.

The AI evaluation includes:

- Overall direction: `improved`, `unchanged`, or `worsened`
- Improvement score from `-5` to `5`
- Improvement magnitude
- Dimension-level scores
- Dimension-level changes
- New information used
- Key improvements
- Remaining issues
- Reasoning summary

The numeric AI score should be treated as a reference signal rather than a definitive measurement. The explanatory fields are usually more important for analysis.

## Tech Stack

- **Python**
- **Streamlit**
- **OpenAI API**
- **Pydantic v2**

## Project Structure

```text
DecisionArticulationLLM/
├── app.py
├── requirements.txt
├── prompts/
│   ├── initial_generation.txt
│   ├── question_generation.txt
│   ├── refinement.txt
│   └── round_improvement_evaluation.txt
├── src/
│   ├── main.py
│   ├── config.py
│   ├── models/
│   │   ├── schemas.py
│   │   └── state.py
│   ├── pipeline/
│   │   ├── initial_generator.py
│   │   ├── question_generator.py
│   │   ├── refiner.py
│   │   ├── improvement_evaluator.py
│   │   └── controller.py
│   ├── evaluation/
│   │   ├── rubric.py
│   │   └── user_rating.py
│   ├── logging/
│   │   ├── logger.py
│   │   └── export.py
│   └── utils/
│       └── llm_client.py
└── data/
    ├── raw/
    ├── processed/
    ├── logs/
    └── evaluations/
```

Some directories are created automatically at runtime if they do not already exist.

## Online Demo

The deployed Streamlit application is available here:

https://decisionarticulationllm-kfpfrparpycckbaymdv3yy.streamlit.app/

Use this link if you only want to test the web interface. No local installation is required for the online version.

## Local Installation

Clone the repository:

```bash
git clone https://github.com/Haonanppp/DecisionArticulationLLM.git
cd DecisionArticulationLLM
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

The project requires an OpenAI API key.

For local use, create a Streamlit secrets file:

```bash
mkdir -p .streamlit
```

Create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your_openai_api_key_here"
```

Do not commit `.streamlit/secrets.toml` to GitHub.

## Running Locally

Run the local Python entry point:

```bash
python -m src.main
```

This starts the command-line version of the decision articulation study.

## Running the Streamlit App Locally

The easiest way to use the Streamlit version is through the deployed link:

https://decisionarticulationllm-kfpfrparpycckbaymdv3yy.streamlit.app/

If you want to run the Streamlit interface locally for development, use:

```bash
streamlit run app.py
```

## Model Configuration

The default model is configured in `src/config.py`.

The Streamlit interface also provides a model selector in the sidebar. The available options include:

- `gpt-5.4`
- `gpt-5.4-mini`
- `Custom`

Use `Custom` if you want to enter another compatible OpenAI model name.

## Data Export

The project can save or export:

- Full study result as JSON
- Round-level summary as CSV
- AI improvement evaluations as JSON
- AI improvement evaluations as CSV

These outputs are useful for analyzing:

- How structured decision quality changes across rounds
- Whether clarification questions add useful information
- Which output dimensions improve most
- Whether users feel better represented after iterative interaction
- Whether AI-based improvement judgments align with user ratings

## Example Use Case

A participant may enter a decision such as:

```text
Decision Title:
Should I apply for a PhD or work after graduation?

Decision Narrative:
I am considering whether to apply for a PhD after finishing my master's degree or directly enter the job market. I enjoy research and complex analytical problems, but I am also concerned about time, stress, financial stability, and long-term career uncertainty.
```

The system will generate an initial structured representation, ask clarification questions, refine the output after the user's answers, and record evaluations across rounds.

## Research Notes

This application is intended to support research on decision articulation and LLM-assisted decision analysis. The goal is not simply to produce a recommendation, but to observe how users express decision context and how iterative AI questioning affects the quality of the resulting decision representation.

Important research considerations include:

- Additional information does not automatically imply improvement.
- A refined output can become worse if it distorts the user's real situation.
- User ratings and AI evaluations should be analyzed together.
- Clarification rounds may show diminishing returns.


## Author

Haonan Pan
