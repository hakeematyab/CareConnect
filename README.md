<!-- PROJECT LOGO -->
<div align="center">
  <a href="https://github.com/hakeematyab/CareConnect/">
    <img src="https://github.com/user-attachments/assets/ae4a4e52-043f-4a59-80d4-4d0952c634c7" alt="Logo" width="250" height="250">
  </a>
</div>

# [CareConnect: A Personal Health Assistant](https://github.com/hakeematyab/CareConnect)
### Atyab Hakeem, Atreyo Das

Increasing wait times for consultations and combating health misinformation, CareConnect aims to offer rapid responses to patient queries, supporting informed decision-making and alleviating the burden on the healthcare system.

## Why CareConnect?

- **Advanced LLMs**: Utilizes cutting-edge large language models to provide accurate and prompt medical advice.
- **Timely Assistance**: Reduces wait times for medical consultations, ensuring health issues are addressed promptly.
- **Combats Misinformation**: Provides reliable information, helping users navigate through health concerns with confidence.
- **Efficient**: Streamlines the process of obtaining medical advice, easing the load on healthcare systems.
- **Informed Decisions**: Empowers users with accurate information, aiding in better health management.

## Ideal for:

- **Patients**: Receive quick and reliable medical guidance to address health concerns without long wait times.
- **Healthcare Systems**: Alleviate the burden on healthcare facilities by providing a first line of advice and information.

**CareConnect** aims to bring efficiency and reliability to medical guidance, ensuring users receive accurate and timely assistance, thereby contributing to a more effective healthcare system.

## Expected Results

CareConnect aims to develop a text generation LLM fine-tuned on a medical question-answering dataset. The model will reliably retrieve relevant factual information and provide appropriate responses, capable of coherently conversing with users to address their queries. The project will compare the performance of various models and address challenges such as low-quality responses, model performance, and latency.


## Demo
<div align="center">
  <a href="https://github.com/hakeematyab/CareConnect/">
    <img src="https://github.com/user-attachments/assets/ac813ed0-de6e-42bb-9def-7649ca04c5ea">
  </a>
</div>

## Performance Metrics

| Metric                | Base LLM |Base LLM with RAG| Fine-tuned LLM | Fine-tuned LLM with RAG  |
|-----------------------|----------|-----------------|----------------|----------------------|
| BERT Precision        | 0.839389 | 0.838776        | 0.838845       | 0.842095             |
| BERT Recall           | 0.824338 | 0.823488        | 0.833013       | 0.836036             |
| BERT F1               | 0.831632 | 0.830919        | 0.835787       | 0.838912             |
| BLEU                  | 0.014211 | 0.010655        | 0.023548       | 0.022102             |
| ROUGE-1               | 0.217740 | 0.212416        | 0.242389       | 0.250984             |
| ROUGE-2               | 0.024242 | 0.022648        | 0.034651       | 0.033410             |
| ROUGE-L               | 0.123481 | 0.120275        | 0.135517       | 0.138729             |
| ROUGE-Lsum            | 0.123884 | 0.121954        | 0.135339       | 0.138489             |
| Semantic Similarity   | 0.459603 | 0.459985        | 0.532590       | 0.545531             |

<!-- 
<div align="left">
  <img src="https://github.com/user-attachments/assets/276f0339-8cfb-4bf0-9f5b-4d06c118f7ff" alt="Performance Comparison Chart" width="600" height="400">
</div>
-->

<!-- SOCIALS -->
## Socials

### Atyab Hakeem
<a href="https://www.linkedin.com/in/hakeem-atyab/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/></a>
<a href="mailto:hakeem.at@northeastern.edu"><img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white"/></a>
<a href="https://github.com/hakeematyab" title="Hakeem Atyab on GitHub">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
</a>

### Atreyo Das
<a href="mailto:das.at@northeastern.edu"><img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white"/></a>
<a href="https://github.com/atreyodas" title="Atreyo Das on GitHub">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
</a>
