<h1 align="center">Theory of Space: Can foundation models construct spatial beliefs through active perception?</h1>
<h3 align="center"><b>🔥 ICLR 2026 🔥</b></h3>

<p align="center" style="font-size: 16px;">
  Pingyue Zhang*, Zihan Huang*, Yue Wang*, Jieyu Zhang*, Letian Xue, Zihan Wang, Qineng Wang, Keshigeyan Chandrasegaran, Ruohan Zhang, Yejin Choi, Ranjay Krishna, Jiajun Wu, Li Fei-Fei, Manling Li
</p>
<p align="center" style="font-size: 12px;"><i>(* equal contribution)</i></p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/📜_Paper-B31B1B?style=for-the-badge&logo=arXiv&logoColor=white" alt="Paper"></a>
  <a href="https://theory-of-space.github.io"><img src="https://img.shields.io/badge/🌐_Website-00C851?style=for-the-badge&logoColor=white" alt="Website"></a>
  <a href="#"><img src="https://img.shields.io/badge/📊_Results-FB8C00?style=for-the-badge&logoColor=white" alt="Results"></a>
</p>

## News
**[2026/01]** Theory of Space is accepted by ICLR 2026

## Introduction
We introduce **Theory of Space (ToS)**, a benchmark evaluating whether foundation models can actively construct spatial beliefs from partial observations. Unlike passive reasoning, ToS requires agents to explore, update, and exploit a globally consistent spatial memory. Current multimodal models struggle with this active, self-directed construction of spatial belief, often relying on passive reasoning from static views.

![image](assets/main.jpg)

## Theory of Space

Theory of Space is the ability to build a mental map from partial views. We define it as three coupled abilities:
*   **Construct**: Actively explore and integrate partial observations into a globally consistent belief.
*   **Update**: Revise the belief when new evidence conflicts with earlier assumptions.
*   **Exploit**: Use the current belief to answer spatial queries and guide the next action.

### Exploration Environment

To construct a spatial belief, an agent must actively explore and integrate partial observations. We use procedurally generated multi-room layouts on N×M grids with paired environments:
*   **Text World**: Symbolic observations with direction/distance bins (pure reasoning).
*   **Vision World**: Egocentric RGB images from ThreeDWorld (perception + reasoning).

The agent uses an action space of **Goto** (move to visible object), **Rotate** (90°/180°/270°), **Observe** (get text/visual view), and **Query** (get coordinates).

### Evaluation Tasks

The agent must use its current belief to answer spatial queries and guide the next action. We evaluate how the learned map is used at two levels:
*   **Route-level** tasks test egocentric, path-based reasoning.
*   **Survey-level** tasks test allocentric, map-like reasoning.

Survey-level probes ask whether a model can infer unseen views and handle geometric transformations beyond memorized paths.

<div align="center">
  <img src="assets/tos_eval.png" alt="Task suite for Theory of Space" />
</div>

#### Results (Active Exploration)
<table>
  <thead>
    <tr>
      <th rowspan="2">Methods</th>
      <th rowspan="2">Avg. step</th>
      <th colspan="5">Route</th>
      <th colspan="4">Survey</th>
      <th rowspan="2">Avg.</th>
    </tr>
    <tr>
      <th>direction</th><th>persp.take</th><th>perc.dec</th><th>act2view</th><th>view2act</th>
      <th>alloc.map</th><th>ment.rot</th><th>loc2view</th><th>view2loc</th>
    </tr>
  </thead>
  <tbody>
    <tr><td colspan="12"><strong>Vision-based World</strong></td></tr>
    <tr>
      <td>GPT-5.2</td>
      <td>17.2</td>
      <td>40.0</td><td><b>36.7</b></td><td>56.2</td><td>43.8</td><td>40.3</td>
      <td>43.4</td><td>59.7</td><td>56.9</td><td>37.8</td>
      <td>46.0</td>
    </tr>
    <tr>
      <td>Gemini-3 Pro</td>
      <td><b>13.6</b></td>
      <td><b>56.3</b></td><td><b>36.7</b></td><td><b>68.2</b></td><td><b>47.2</b></td><td><b>54.0</b></td>
      <td><b>63.5</b></td><td><b>73.0</b></td><td><b>65.4</b></td><td><b>52.2</b></td>
      <td><b>57.3</b></td>
    </tr>
    <tr>
      <td>Claude-4.5 Sonnet</td>
      <td>19.6</td>
      <td>23.5</td><td>23.7</td><td>12.7</td><td>32.7</td><td>20.7</td>
      <td>37.8</td><td>34.0</td><td>39.5</td><td>34.2</td>
      <td>32.1</td>
    </tr>
    <tr>
      <td>GLM-4.6V</td>
      <td>15.0</td>
      <td>15.5</td><td>18.5</td><td>3.3</td><td>14.5</td><td>0.7</td>
      <td>18.9</td><td>7.3</td><td>31.6</td><td>18.8</td>
      <td>16.0</td>
    </tr>
    <tr>
      <td>Qwen3-VL</td>
      <td>16.3</td>
      <td>17.2</td><td>23.7</td><td>13.7</td><td>23.8</td><td>6.0</td>
      <td>26.1</td><td>17.0</td><td>21.8</td><td>43.7</td>
      <td>21.4</td>
    </tr>
    <tr><td colspan="12"><strong>Text-based World</strong></td></tr>
    <tr>
      <td>GPT-5.2</td>
      <td><b>11.4</b></td>
      <td>68.8</td><td>70.5</td><td>80.3</td><td>71.0</td><td>53.7</td>
      <td>77.9</td><td>81.0</td><td>79.1</td><td>66.0</td>
      <td>72.0</td>
    </tr>
    <tr>
      <td>Gemini-3 Pro</td>
      <td>13.5</td>
      <td><b>78.0</b></td><td><b>79.2</b></td><td><b>90.6</b></td><td><b>75.3</b></td><td><b>76.3</b></td>
      <td><b>81.0</b></td><td><b>94.0</b></td><td><b>83.3</b></td><td><b>76.2</b></td>
      <td><b>81.5</b></td>
    </tr>
    <tr>
      <td>Claude-4.5 Sonnet</td>
      <td>18.7</td>
      <td>66.0</td><td>65.0</td><td>78.9</td><td>62.0</td><td>53.0</td>
      <td>69.2</td><td>76.3</td><td>67.2</td><td>56.5</td>
      <td>68.8</td>
    </tr>
    <tr>
      <td>GLM-4.6V</td>
      <td>14.5</td>
      <td>20.5</td><td>19.7</td><td>12.0</td><td>15.2</td><td>3.7</td>
      <td>13.6</td><td>9.3</td><td>26.1</td><td>22.5</td>
      <td>18.9</td>
    </tr>
    <tr>
      <td>InternVL-3.5</td>
      <td>15.0</td>
      <td>29.3</td><td>45.3</td><td>26.3</td><td>36.3</td><td>8.0</td>
      <td>30.7</td><td>27.3</td><td>39.1</td><td>34.0</td>
      <td>33.5</td>
    </tr>
    <tr>
      <td>Qwen3-VL</td>
      <td>14.1</td>
      <td>33.7</td><td>48.0</td><td>50.7</td><td>36.0</td><td>9.3</td>
      <td>36.0</td><td>35.3</td><td>35.3</td><td>47.1</td>
      <td>36.8</td>
    </tr>
  </tbody>
</table>


#### Results (Passive Understanding)
<table>
  <thead>
    <tr>
      <th rowspan="2">Methods</th>
      <th colspan="5">Route</th>
      <th colspan="4">Survey</th>
      <th rowspan="2">Avg.</th>
    </tr>
    <tr>
      <th>direction</th><th>persp.take</th><th>perc.dec</th><th>act2view</th><th>view2act</th>
      <th>alloc.map</th><th>ment.rot</th><th>loc2view</th><th>view2loc</th>
    </tr>
  </thead>
  <tbody>
    <tr><td colspan="11"><strong>Vision-based World</strong></td></tr>
    <tr>
      <td>GPT-5.2</td>
      <td>47.3</td><td>35.0</td><td><b>63.9</b></td><td><b>54.5</b></td><td>49.3</td>
      <td>64.8</td><td>83.3</td><td>50.3</td><td>65.6</td>
      <td>57.1</td>
    </tr>
    <tr>
      <td>Gemini-3 Pro</td>
      <td><b>63.8</b></td><td><b>36.3</b></td><td>57.5</td><td>49.0</td><td><b>58.0</b></td>
      <td><b>67.2</b></td><td><b>85.3</b></td><td><b>70.4</b></td><td><b>57.0</b></td>
      <td><b>60.5</b></td>
    </tr>
    <tr>
      <td>Claude-4.5 Sonnet</td>
      <td>48.0</td><td>33.5</td><td>38.5</td><td>39.5</td><td>16.7</td>
      <td>54.9</td><td>58.3</td><td>54.9</td><td>44.7</td>
      <td>44.8</td>
    </tr>
    <tr>
      <td>GLM-4.6V</td>
      <td>11.2</td><td>24.0</td><td>4.7</td><td>18.5</td><td>3.3</td>
      <td>22.5</td><td>11.7</td><td>20.0</td><td>33.4</td>
      <td>18.0</td>
    </tr>
    <tr>
      <td>Qwen3-VL</td>
      <td>20.7</td><td>28.8</td><td>23.1</td><td>24.3</td><td>4.7</td>
      <td>33.2</td><td>21.3</td><td>27.2</td><td>40.9</td>
      <td>24.9</td>
    </tr>
    <tr><td colspan="11"><strong>Text-based World</strong></td></tr>
    <tr>
      <td>GPT-5.2</td>
      <td><b>84.5</b></td><td>88.2</td><td><b>97.0</b></td><td><b>89.0</b></td><td><b>76.0</b></td>
      <td><b>96.3</b></td><td><b>98.3</b></td><td><b>94.8</b></td><td><b>89.2</b></td>
      <td><b>90.4</b></td>
    </tr>
    <tr>
      <td>Gemini-3 Pro</td>
      <td>82.7</td><td><b>92.7</b></td><td><b>97.0</b></td><td>87.5</td><td><b>75.7</b></td>
      <td>86.2</td><td>91.3</td><td>85.7</td><td>80.0</td>
      <td>86.5</td>
    </tr>
    <tr>
      <td>Claude-4.5 Sonnet</td>
      <td>74.2</td><td>81.5</td><td>90.6</td><td>70.8</td><td>60.0</td>
      <td>76.8</td><td>74.3</td><td>70.6</td><td>58.5</td>
      <td>75.4</td>
    </tr>
    <tr>
      <td>GLM-4.6V</td>
      <td>22.2</td><td>40.5</td><td>25.4</td><td>24.8</td><td>4.3</td>
      <td>21.2</td><td>9.7</td><td>36.2</td><td>26.8</td>
      <td>27.1</td>
    </tr>
    <tr>
      <td>InternVL-3.5</td>
      <td>36.0</td><td>67.7</td><td>43.0</td><td>42.5</td><td>8.7</td>
      <td>37.5</td><td>18.7</td><td>44.5</td><td>38.3</td>
      <td>39.5</td>
    </tr>
    <tr>
      <td>Qwen3-VL</td>
      <td>40.3</td><td>68.2</td><td>55.2</td><td>43.5</td><td>17.3</td>
      <td>43.0</td><td>40.0</td><td>42.3</td><td>54.9</td>
      <td>45.0</td>
    </tr>
  </tbody>
</table>


### Probing Spatial Belief

We probe the agent's internal belief state to understand *why* failures occur. We provide a direct window into the agent's spatial belief via **explicit cognitive-map probing**. The agent outputs a structured cognitive map (N×M grid) at each step.

<div align="center">
  <img src="assets/spatial_belief.png" alt="Spatial belief probing" />
</div>

#### Metrics
*   **Correctness (final)**: Evaluates the predicted *global* map at the last turn.
*   **Perception**: Compares the predicted *local* map to the ground-truth local map for the current field of view.
*   **Self-tracking**: Compares the agent pose inferred from the predicted global map to the ground-truth agent state.
*   **Local ↔ Global**: Compares local and global predictions *within the same turn* (coherence check).
*   **Stability**: Checks if previously observed objects degrade in the map over time.
*   **Uncertainty**: Can the agent identify which regions it hasn't seen yet?

#### Results
<table>
  <thead>
    <tr>
      <th rowspan="2">Methods</th>
      <th colspan="2">Correctness</th>
      <th colspan="2">Perception</th>
      <th colspan="2">Local ↔ Global</th>
      <th colspan="2">Stability</th>
      <th colspan="2">Self-tracking</th>
      <th rowspan="2">Uncertainty</th>
    </tr>
    <tr>
      <th>Ori.</th><th>Overall</th>
      <th>Ori.</th><th>Pos.</th>
      <th>Ori.</th><th>Pos.</th>
      <th>Ori.</th><th>Pos.</th>
      <th>Ori.</th><th>Pos.</th>
    </tr>
  </thead>
  <tbody>
    <tr><td colspan="12"><strong>Vision-based World</strong></td></tr>
    <tr>
      <td>GPT-5.2</td>
      <td>20.2</td><td>32.2</td>
      <td>33.5</td><td><b>72.4</b></td>
      <td><b>57.9</b></td><td>58.7</td>
      <td><b>65.4</b></td><td>56.4</td>
      <td>93.3</td><td>64.7</td>
      <td>53.7</td>
    </tr>
    <tr>
      <td>GEMINI-3 PRO</td>
      <td><b>32.2</b></td><td><b>52.1</b></td>
      <td><b>43.8</b></td><td>68.5</td>
      <td>52.9</td><td><b>68.3</b></td>
      <td>61.8</td><td><b>62.0</b></td>
      <td><b>98.8</b></td><td><b>73.9</b></td>
      <td><b>68.8</b></td>
    </tr>
    <tr><td colspan="12"><strong>Text-based World</strong></td></tr>
    <tr>
      <td>GPT-5.2</td>
      <td>91.0</td><td>80.0</td>
      <td><b>100</b></td><td>86.8</td>
      <td><b>96.4</b></td><td><b>86.0</b></td>
      <td><b>96.7</b></td><td>67.6</td>
      <td>98.0</td><td>86.7</td>
      <td>64.5</td>
    </tr>
    <tr>
      <td>GEMINI-3 PRO</td>
      <td><b>92.5</b></td><td><b>81.4</b></td>
      <td>99.9</td><td><b>88.2</b></td>
      <td>91.6</td><td>84.8</td>
      <td>90.8</td><td><b>67.7</b></td>
      <td><b>99.9</b></td><td><b>85.2</b></td>
      <td><b>79.2</b></td>
    </tr>
  </tbody>
</table>

#### Probing Conclusion
*   **Perception is the bottleneck**: Vision perception remains a key bottleneck, especially for object orientation.
*   **Unstable belief**: Unstable cognitive map prediction degrades spatial belief beyond initial perception.

### Belief Update

An agent must revise its belief when new evidence conflicts with earlier assumptions. We introduce a dynamic perturbation task to probe **Belief Update**. After exploration, objects are secretly relocated, creating a "false belief" that conflicts with new observations. The agent must actively re-explore to identify changes and revise its map.

#### Results
<table>
  <thead>
    <tr>
      <th rowspan="2">Methods</th>
      <th colspan="2">Avg. Steps</th>
      <th colspan="2">Identification</th>
      <th colspan="2">Belief Correctness</th>
      <th colspan="2">Belief Update</th>
    </tr>
    <tr>
      <th>All</th><th>Red.</th>
      <th>Ori.</th><th>Pos.</th>
      <th>Ori.</th><th>Pos.</th>
      <th>Ori.</th><th>Pos.</th>
    </tr>
  </thead>
  <tbody>
    <tr><td colspan="9"><strong>Vision-based World</strong></td></tr>
    <tr>
      <td>GPT-5.2</td>
      <td>8.64</td><td>2.94</td>
      <td>13.0</td><td>63.6</td>
      <td>12.5</td><td>40.7</td>
      <td>15.5</td><td>30.8</td>
    </tr>
    <tr>
      <td>GEMINI-3 PRO</td>
      <td><b>7.30</b></td><td><b>1.84</b></td>
      <td><b>25.5</b></td><td><b>78.1</b></td>
      <td><b>30.2</b></td><td><b>68.1</b></td>
      <td><b>26.4</b></td><td><b>43.4</b></td>
    </tr>
    <tr><td colspan="9"><strong>Text-based World</strong></td></tr>
    <tr>
      <td>GPT-5.2</td>
      <td><b>6.22</b></td><td>0.23</td>
      <td><b>96.3</b></td><td><b>95.5</b></td>
      <td>88.0</td><td><b>75.5</b></td>
      <td><b>93.4</b></td><td><b>45.7</b></td>
    </tr>
    <tr>
      <td>GEMINI-3 PRO</td>
      <td>6.26</td><td><b>0.14</b></td>
      <td>95.9</td><td>94.1</td>
      <td><b>89.4</b></td><td>71.2</td>
      <td>89.1</td><td>42.8</td>
    </tr>
  </tbody>
</table>

### Key Findings

#### 01. Active Exploration Bottleneck
Active exploration is a key bottleneck: performance drops when models must choose actions under partial observability, compared to passive comprehension from standardized logs.

<p align="center">
  <img src="assets/passive-active-gap-text.png" width="45%" alt="Passive vs active gap in text world" />
  <img src="assets/passive-active-gap-vision.png" width="45%" alt="Passive vs active gap in vision world" />
</p>
<p align="center">
  <img src="assets/infogain.png" width="60%" alt="Information gain analysis" />
</p>

#### 02. Modality Gap
A clear modality gap persists: text-based settings consistently outperform vision-based settings in spatial belief construction and utilization.

<p align="center">
  <img src="assets/vision-text-gap-passive.png" width="45%" alt="Vision vs text gap under passive setting" />
  <img src="assets/vision-text-gap-active.png" width="45%" alt="Vision vs text gap under active setting" />
</p>

#### 03. Active-Passive Gap Increases with Complexity
The active–passive gap increases with complexity; Gemini-3 Pro scales much better. As the number of rooms increases, exploration cost rises accordingly. For both GPT-5.2 and Gemini-3 Pro, performance declines as the room number increases, and the active–passive performance gap widens with room number.

<table>
  <thead>
    <tr>
      <th rowspan="2">Methods</th>
      <th colspan="3">2-room</th>
      <th colspan="3">4-room</th>
    </tr>
    <tr>
      <th>pass.</th><th>act.</th><th>exp.</th>
      <th>pass.</th><th>act.</th><th>exp.</th>
    </tr>
  </thead>
  <tbody>
    <tr><td colspan="7"><strong>Text-based World</strong></td></tr>
    <tr>
      <td>GPT-5.2</td>
      <td>92.3</td><td>77.8</td><td>6.2</td>
      <td>86.5</td><td>66.0</td><td>16.4</td>
    </tr>
    <tr>
      <td>Gemini-3 Pro</td>
      <td>86.7</td><td>80.6</td><td>6.2</td>
      <td>81.2</td><td>77.7</td><td>19.7</td>
    </tr>
    <tr><td colspan="7"><strong>Vision-based World</strong></td></tr>
    <tr>
      <td>GPT-5.2</td>
      <td>59.3</td><td>51.5</td><td>10.8</td>
      <td>52.6</td><td>40.3</td><td>23.2</td>
    </tr>
    <tr>
      <td>Gemini-3 Pro</td>
      <td>58.3</td><td>57.8</td><td>6.6</td>
      <td>56.2</td><td>51.5</td><td>19.7</td>
    </tr>
  </tbody>
</table>

#### 04. Bottleneck: Perception & Stability
Vision perception remains a key bottleneck, especially for object orientation. Unstable cognitive map prediction degrades spatial belief beyond initial perception.

## Usage

### Setup (run everything)
Run `setup.sh` to install deps, download data into `room_data/3-room/`, and run the default experiment (a subset of 10 runs). You can set API keys in the script.
```bash
bash setup.sh
```

### Model configuration

To add a new model, edit `scripts/SpatialGym/base_model_config.yaml` and add an entry under the `models` section. Each model requires specific parameters based on its provider:

#### OpenAI Models
```yaml
models:
  gpt-5.2:
    provider: openai # API provider (`openai`, `anthropic`)
    model_name: gpt-5.2 # Model identifier used by the API
    max_completion_tokens: 32768 # Maximum tokens for response
    temperature: 1.0 # Sampling temperature
    max_workers: 128 # Maximum parallel API calls
    max_retries: 5 # Retry rounds for batch requests
    max_retries_api: 5 # Retry attempts per API request
    timeout: 500 # Request timeout in seconds
    reasoning_effort: medium # Reasoning effort (low, medium, high)
```

#### VLLM Models
First run the following command to serve the model:
```bash
vllm serve Qwen/Qwen3-VL-2B-Instruct \
  --host 0.0.0.0 \
  --port 9999 \
  --dtype bfloat16 \
  --served-model-name qwen3-vl-2b-instruct \
  --max_model_len 128000 \
```

Then add the following entry to `scripts/SpatialGym/base_model_config.yaml`:
```yaml
  qwen3-vl-2b-instruct:
    provider: openai
    organization: self-hosted
    model_name: qwen3-vl-2b-instruct # same as served-model-name in vllm serve command
    base_url: http://localhost:9999/v1 # same as host and port in vllm serve command
    max_completion_tokens: 8192
    temperature: 0.0
    max_workers: 16
    max_retries: 3
    timeout: 600
```
  
For other models, you can refer to the `scripts/SpatialGym/base_model_config.yaml` for more details.

### Commands
Run a single full pipeline (explore + eval + optional cogmap):
```bash
python scripts/SpatialGym/spatial_run.py \
  --phase all \
  --model-name gpt-5.2 \
  --num 25 \ # 25 samples (run00 - run24)
  --data-dir room_data/3-room/ \
  --output-root result/ \
  --render-mode vision,text \ # run both vision and text world
  --exp-type active,passive \ # run both active and passive exploration
  --cogmap \ # run cognitive map evaluation
  --false-belief-exp \ # run false belief experiment
```
See `scripts/SpatialGym/README.md` for more commands.


### Output Structure

Results are organized as:
```
{output_root}/
└─ {model_name}/
   └─ {room_hash}/
      └─ {render_mode}/
         └─ {exp_type}/
            └─ {think|nothink}/
               ├─ config.json
               ├─ exploration.json
               ├─ evaluation.json
               ├─ images/
               └─ [proxy_agent]/            # only when exp_type = passive
                  ├─ config.json
                  ├─ exploration.json
                  ├─ evaluation.json
                  └─ images/
```

### Visulization
You can find the visulization html under result/gpt-5.2/env_data.html

The main page will have explore + eval + optional cogmap + false belief + correlation metrics.
<p align="center">
  <img src="assets/visualization.png" width="45%" alt="Visulization of all samples" />
  <img src="assets/charts.png" width="45%" alt="Visulization charts of all samples" />
</p>

Each sample page features comprehensive data for each turn, along with relevant sample metrics.
<p align="center">
  <img src="assets/visualization_sample.png" width="60%" alt="Visulization of each sample" />
</p>

---

## Visual Scene Generation (ToS-vision-scenes)

The `ToS-vision-scenes/` submodule provides tools to generate custom multi-room 3D environments using TDW (ThreeDWorld). Use this if you want to create your own visual scene datasets.

> **Remote Server Requirement**: If running on a remote server, make sure the server has a **graphical display** attached (not headless). TDW and Unity require a display for rendering.

### Pre-generated Datasets

We provide pre-generated visual scene datasets on Hugging Face:

| Dataset | Rooms | Runs | Link |
|---------|-------|------|------|
| 2-room scenes | 2 | 25 | [tos_dataset_0103_2room_25runs](https://huggingface.co/datasets/yw12356/tos_dataset_0103_2room_25runs) |
| 3-room scenes | 3 | 100 | [tos_dataset_0127_3room_100runs](https://huggingface.co/datasets/yw12356/tos_dataset_0127_3room_100runs) |
| 4-room scenes | 4 | 25 | [tos_dataset_0103_4room_25runs](https://huggingface.co/datasets/yw12356/tos_dataset_0103_4room_25runs) |

> **Note**: The 3-room dataset includes false-belief experiment data (`falsebelief_exp.json`) for each run, used for belief updating evaluation.

### Features
- Procedural room layout generation
- Custom 3D model support with asset bundle building
- False-belief experiment scene generation
- Pre-render validation for spatial tasks

### Quick Start

```bash
# Clone with submodule
git clone --recursive https://github.com/williamzhangNU/Theory-of-Space.git

# Or update submodule if already cloned
git submodule update --init --recursive

# Setup (after running main setup.sh)
cd ToS-vision-scenes
source setup.sh
```

See [`ToS-vision-scenes/README.md`](ToS-vision-scenes/README.md) for full documentation.