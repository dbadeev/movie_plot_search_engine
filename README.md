# Project movie_plot_search_engine
This project uses a multi-agent architecture for accurate movie recommendations. 
AI agents will analyze it, search a database of over 500,000 films, and return the top 3 matches 
with expert justifications based on user's story description.

## Key Features
* __Multi-Agent System__: Specialized AI agents for text editing, analysis, and expert recommendation.
* __High-Performance Search__: Semantic search using FAISS for instant results.
* __Expert Justifications__: Each recommendation comes with a detailed, AI-generated explanation.
* __Interactive UI__: A simple and responsive web interface powered by Gradio.

## How It Works
1. __You Describe a Plot__: Enter a 50-100 word story description in the Gradio interface.
2. __*Editor Agent* Refines It__: An AI agent validates, cleans, and improves your text for clarity and grammar.
3. __*Critic Agent* Creates an Overview__: A second agent generates a professional, IMDb-style synopsis based on 
your story description.
4. __System Finds Matches__: The overview is converted into a vector embedding and compared against 500,000+ movies using FAISS.
5. __*Expert Agent* Selects & Explains__: A third agent analyzes the top candidates, selects the best three, and writes a unique justification for each.
6. __You Get Recommendations__: The results are displayed in clean, formatted form.

## Getting Started
### Prerequisites
* Python 3.11+
* A Modal account and authenticated CLI.
* A Nebius AI Studio API key.

### Installation & Launch
1. Clone the Repository:
```
bash

git clone https://github.com/your-repo/movie-plot-search-engine.git
cd movie-plot-search-engine
```
2. Install Local Dependencies:
```
bash

pip install -r requirements_local.txt
```

3. Deploy and Run:
```
bash

python movie_plot_search_engine.py
```
### Example
[Program's operation example](https://drive.google.com/file/d/1ze3mG20PBl1q7WGFaqCghkBYzubBirzK/view?usp=drive_link)

More detailed description of the pipeline (in Russian) is in the [wiki]().
<br><br>

## Author
*Dmitry Badeev* 

### License
This project is licensed under the MIT License.
