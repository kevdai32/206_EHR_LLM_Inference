# 206_EHR_LLM_Inference

## Drug of Interest
**Imatinib (Gleevec)**

---

## Purpose
The objective of this cohort is to assess the effectiveness of **imatinib (Gleevec)**, a first-generation Tyrosine Kinase Inhibitor (TKI) designed specifically for patients diagnosed with Chronic Myeloid Leukemia (CML), utilizing data from the UCSF/SPHD dataset.

---

## Background
**Chronic Myeloid Leukemia (CML)** is a cancer that originates in the blood-forming cells of the bone marrow.  
It is characterized by the overproduction of abnormal granulocytes (a type of white blood cell). These abnormal cells, referred to as *blasts* in advanced stages, can overcrowd the bone marrow and bloodstream, leading to complications such as:

- Anemia  
- Increased risk of infection  
- Easy bleeding  

The defining feature of CML is the presence of the **Philadelphia chromosome**, a genetic abnormality caused by a translocation between chromosomes 9 and 22. This creates the **BCR-ABL fusion gene**, which produces a constitutively active tyrosine kinase enzyme that drives uncontrolled proliferation of leukemic cells.  

CML often progresses through stages:

1. **Chronic phase** – frequently asymptomatic.  
2. **Accelerated phase** – more pronounced symptoms.  
3. **Blast crisis** – resembles acute leukemia with fatigue, weakness, bleeding, and frequent infections.  

**Treatment**:  
- Tyrosine kinase inhibitors (TKIs) are the mainstay of therapy, blocking the activity of BCR-ABL.  
- **Allogeneic bone marrow transplantation (BMT)** may be considered for patients with resistance, intolerance, or advanced disease, especially in younger patients with suitable donors.  

**Imatinib (Gleevec):**
- The first TKI developed for CML.  
- Significantly improved prognosis, transforming CML from a fatal disease into a manageable chronic condition.  
- Remains a widely used **first-line therapy**, particularly for patients with low-risk disease or those who tolerate it well.  
- Newer TKIs are available for patients with resistance or intolerance, but imatinib continues to be highly relevant.

---

## Cohort Criteria

### Inclusion
- Newly confirmed diagnosis of **Philadelphia chromosome–positive (Ph+) CML** in the chronic phase.  
- Age ≥ 18 (CML is rare in younger populations).  

### Exclusion
- Prior use of TKIs (including imatinib) before diagnosis.  
- Use of TKIs other than imatinib after diagnosis.  
- Prior history of **bone marrow transplant (BMT)**.  
- History of **accelerated or blast phase CML** at or prior to treatment initiation.  
- Patients are tracked for up to **10 years** to evaluate survival, remission, or until death/end of observation.

### References
References: 
- Iqbal N, Iqbal N. Imatinib: a breakthrough of targeted therapy in cancer. Chemother Res Pract. 2014;2014:357027. doi: 10.1155/2014/357027. Epub 2014 May 19. PMID: 24963404; PMCID: PMC4055302. 
- Hochhaus A, Larson RA, Guilhot F, Radich JP, Branford S, Hughes TP, Baccarani M, Deininger MW, Cervantes F, Fujihara S, Ortmann C‑E, Menssen HD, Kantarjian H, O’Brien SG, Druker BJ; IRIS Investigators. Long‑Term Outcomes of Imatinib Treatment for Chronic Myeloid Leukemia. New England Journal of Medicine. 2017 Mar 9;376(10):917–927. doi:10.1056/NEJMoa1609324  

## Recommended System Requirements
### All notebooks in this repository were run using AWS
Running the notebooks for getting notes the following hardware is recommended

- **CPU**: 8+ cores recommended  
- **RAM**: 61+ GB system memory min. It is highly recommended to aim for double the amount of ram ~120gb 
- **Disk**: ~10 GB free space for model checkpoints and intermediate outputs  
- **Environment**: Python 3.10+ with the following libraries:
  - `duckdb`  
  - `pandas`  
  - `json` (built-in)  
  - `numpy`

# Clinical notes
- Clinical notes were obtained using SQL from ATLAS. Please look into `Get_Notes.ipynb` to see code to grab the latest `Progress Notes` for each patient.
- I selected `Progress_notes` because they allow me to track each patient after diagnosis, determine whether imatinib was prescribed as the sole treatment, and assess whether its prescription ultimately led to disease regression.

# Table 1. Demographics
## Demographics Extraction
All demographics were extracted by taking the `person.id` column from `latest_note.parquet` and querying it in UCSF **Emerse**.
| Demographics | n = 1762 |
|--------------|--|
| **Gender** |  |
| Male |1018  |
| Female |742  |
| **Race** |N/A  |
| White |1013  |
| Unknown |6  |
| Other Race |206  |
| Asian |108  |
| Native Hawaiian or Other Pacific Islander |98  |
| Black or African American |86  |
| American Indian or Alaska Native |4  |
| Declined |11  |
| **Ethnicity** |N/A  |
| Not Hispanic or Latino |1266  |
| Unspecified |175  |
| Declined |13  |
| Unknown|11  |
| Unknown/declined |146  |
| **Age** | N/A |
| Avg |73  |
| Median |74.5 |
| Max |99  |
| Min |20  |
| STDev |15.6  |
*Note: Age statistics were approximated by taking the midpoint of each age range and weighting by the number of patients in that range.*

 - The demographics above closely mirror those observed in Atlas. After filtering for progress notes only, the total patient count decreased from 1,936 to 1,762. One possible explanation is that these patients are new to UCSF and therefore do not yet have progress notes, but may instead have other types of documentation such as consultation notes.
   
# Regular Expression
- Please run `Regular_expression.ipynb` on `latest_notes.parquet` to reproduce the results and grab `source_person_values` to grab demographics in [UCSF Emerse](https://emerse.ucsf.edu/) 

# Table 2. Regular Expression 
| Concept                 | True | False |
|-------------------------|------|-------|
| cml_diagnosed           | 248  | 1514  |
| aml_diagnosed           | 82   | 1680  |
| blast_phase_cml         | 7    | 1755  |
| acute_phase_cml         | 0    | 1762  |
| bmt_history             | 181  | 1581  |
| imatinib_mentioned      | 191  | 1571  |
| related_drugs_mentioned | 170  | 1592  |

# Table 3. Regular Expression imatinib mention demographics
| Demographics | n = 191 |
|--------------|--|
| **Gender** |  |
| Male |115  |
| Female |76  |
| **Race** |N/A  |
| White |85  |
| Unknown |16  |
| Other Race |27  |
| Asian |12  |
| Native Hawaiian or Other Pacific Islander |10  |
| Black or African American |11  |
| American Indian or Alaska Native |1  |
| Declined |2  |
| **Ethnicity** |N/A  |
| Not Hispanic or Latino |119  |
|Hispanic or Latino|20|
| Unspecified |28  |
| Declined |5  |
| Unknown|2 |
| Unknown/declined |17  |
| **Age** | N/A |
| Avg |65.3  |
| Median |64.5|
| Max |94.5  |
| Min |34.5  |
| STDev |16.5 |


# LLM Inference
## Reproducibility

To replicate these results please run the LLM notebooks located in the [`Note_book`](./Note_book) folder.

- `DeepSeek-R1-14b(1).ipynb` – Runs DeepSeek-R1-14b model 
- `LLM_inference(1).ipynb` – Runs openhermes-2.5-mistral-7b.Q2_K  
- `MEDGEMMA_4B(1).ipynb` – Runs Medgemma_4B
  
## Recommended System Requirements

Running the notebooks in this project requires a machine with sufficient compute resources for LLM inference:

- **GPU**: NVIDIA GPU with at least 24 GB VRAM (e.g., RTX 3090, RTX 4090, A100, or L4)  
- **CPU**: 8+ cores recommended  
- **RAM**: 32 GB system memory  
- **Disk**: ~20 GB free space for model checkpoints and intermediate outputs  
- **Environment**: Python 3.10+, PyTorch 2.0+, Hugging Face `transformers` and `accelerate`  

## LLM Models selected
| Model Name                                 | Type                  | Size  | Notes                                                                 | Hugging Face Link                                                             |
|--------------------------------------------|-----------------------|-------|-----------------------------------------------------------------------|--------------------------------------------------------------------------------|
| openhermes-2.5-mistral-7b.Q2_K              | Instruction / Chat    | 7 B   | Quantized version of OpenHermes 2.5 (Mistral 7B fine-tune)            | [TheBloke/OpenHermes-2.5-Mistral-7B-GGUF]                                     |
| Deep-Seek-R1-Distill-Qwen-14B               | Reasoning / Chat      | 14 B  | Distilled from DeepSeek-R1 with strong reasoning and “chain-of-thought” behavior | [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B]                                   |
| MEDGEMMA-4B-it                             | Medical / Multimodal  | 4 B   | Instruction-tuned Gemma variant optimized for healthcare text and images | [google/medgemma-4b-it]                                                       |
- For each model I used the same system prompt:
## System Prompt

```yaml
You are a strict JSON generator. Only output JSON that matches the schema below.
Do not include any extra text, commentary, or explanations.

Schema:
- imatinib_mentioned: true if the drug imatinib (also known as Gleevec) is mentioned in the note, otherwise false
- related_drugs_mentioned: true if drugs related to imatinib (e.g., dasatinib, nilotinib, bosutinib) are mentioned, otherwise false
- cml_diagnosed: true if chronic myeloid leukemia (CML) is diagnosed, otherwise false
- cml_in_regression: true if chronic myeloid leukemia is mentioned as being in regression, otherwise false
- aml_diagnosed: true if acute myeloid leukemia (AML) is diagnosed, otherwise false
- blast_phase_cml: true if blast phase CML is explicitly mentioned, otherwise false
- bmt_history: true if history of bone marrow transplant (BMT) is mentioned, otherwise false
- acute_phase_cml: true if acute phase CML is explicitly mentioned, otherwise false

Rules:
1. Only mark a field as true if the note clearly indicates it.
2. If the note does not explicitly mention a field, mark it false.
3. The output must always be valid JSON with all eight fields present.
```

# Table 4. MedGemma-LLM-Inference-Results
| Model                | Field                   | True | False | 
|----------------------|--------------------------|------|-------|
| Google-MedGemma-4b   | imatinib_mentioned       | 316  | 1446  | 
| Google-MedGemma-4b   | related_drugs_mentioned  | 487  | 1275  | 
| Google-MedGemma-4b   | cml_diagnosed            | 781  | 981   |
| Google-MedGemma-4b   | cml_in_regression        | 435  | 1327  |
| Google-MedGemma-4b   | aml_diagnosed            | 35   | 1727  | 
| Google-MedGemma-4b   | blast_phase_cml          | 18   | 1744  | 
| Google-MedGemma-4b   | bmt_history              | 322  | 1440  | 
| Google-MedGemma-4b   | acute_phase_cml          | 34   | 1728  | 

# Table 5. Med-Gemma imatinib mention demographics
| Demographics | n = 316 |
|--------------|--|
| **Gender** |  |
| Male |187  |
| Female |129  |
| **Race** |N/A  |
| White |163  |
| Unknown |24  |
| Other Race |34  |
| Asian |11  |
| Native Hawaiian or Other Pacific Islander |N/A  |
| Black or African American |15  |
| American Indian or Alaska Native |N?A  |
| Declined |24  |
| **Ethnicity** |N/A  |
| Not Hispanic or Latino |199  |
|Hispanic or Latino|27|
| Unspecified |49  |
| Declined |3  |
| Unknown|2 |
| Unknown/declined |36  |
| **Age** | N/A |
| Avg |67.2  |
| Median |64.5|
| Max |94.5  |
| Min |34.5  |
| STDev |15.5| 


# Table 6. OpenHermes-LLM-Inference-Results
| Model              | Field                  | True | False | 
|--------------------|------------------------|------|-------|
| OpenHermes-2.5-7B  | imatinib_mentioned     | 328  | 1434  |
| OpenHermes-2.5-7B  | related_drugs_mentioned| 371  | 1391  | 
| OpenHermes-2.5-7B  | cml_diagnosed          | 524  | 1238  | 
| OpenHermes-2.5-7B  | cml_in_regression      | 351  | 1411  | 
| OpenHermes-2.5-7B  | aml_diagnosed          | 145  | 1617  | 
| OpenHermes-2.5-7B  | blast_phase_cml        | 75   | 1687  | 
| OpenHermes-2.5-7B  | bmt_history            | 381  | 1381  | 
| OpenHermes-2.5-7B  | acute_phase_cml        | 92   | 1670  | 

# Table 7. Open-Hermes imatinib mention demographics
| Demographics | n = 328 |
|--------------|--|
| **Gender** |  |
| Male |206  |
| Female |122  |
| **Race** |N/A  |
| White |152  |
| Unknown |62  |
| Other Race |52  |
| Asian |19  |
| Native Hawaiian or Other Pacific Islander |16  |
| Black or African American |20  |
| American Indian or Alaska Native |1  |
| Declined |22  |
| **Ethnicity** |N/A  |
| Not Hispanic or Latino |206  |
|Hispanic or Latino|41|
| Unspecified |47  |
| Declined |8  |
| Unknown|2 |
| Unknown/declined |24  |
| **Age** | N/A |
| Avg |70.5  |
| Median |74.5|
| Max |94.5  |
| Min |34.5  |
| STDev |15.7 |


# Table 8. DeepSeek-R1-14b-LLM-Inference-Results
| Model            | Field                   | True | False | 
|------------------|-------------------------|------|-------|
| DeepSeek_R1_14b  | imatinib_mentioned      | 191  | 1571  | 
| DeepSeek_R1_14b  | related_drugs_mentioned | 186  | 1576  | 
| DeepSeek_R1_14b  | cml_diagnosed           | 257  | 1505  | 
| DeepSeek_R1_14b  | cml_in_regression       | 24   | 1738  | 
| DeepSeek_R1_14b  | aml_diagnosed           | 99   | 1663  |
| DeepSeek_R1_14b  | blast_phase_cml         | 26   | 1736  | 
| DeepSeek_R1_14b  | bmt_history             | 195  | 1567  | 
| DeepSeek_R1_14b  | acute_phase_cml         | 26   | 1736  |

# Table 9. DeepSeek imatinib mention demographics
| Demographics | n = 191 |
|--------------|--|
| **Gender** |  |
| Male |115  |
| Female |76  |
| **Race** |N/A  |
| White |84  |
| Unknown |16  |
| Other Race |28  |
| Asian |12  |
| Native Hawaiian or Other Pacific Islander |10  |
| Black or African American |11  |
| American Indian or Alaska Native |1  |
| Declined |29  |
| **Ethnicity** |N/A  |
| Not Hispanic or Latino |119  |
|Hispanic or Latino|20|
| Unspecified |28  |
| Declined |5  |
| Unknown|2 |
| Unknown/declined |17  |
| **Age** | N/A |
| Avg |60  |
| Median |645|
| Max |94.5  |
| Min |34.5  |
| STDev |16.5 |

# Comparision with Manual Notes

# Table 10. % True by Field Across Models (300 Notes)

| Field                   | Google-MedGemma-4b | OpenHermes-2.5-7B | DeepSeek_R1_14b | Manual |
|-------------------------|------------------|------------------|----------------|--------|
| imatinib_mentioned      | 17.7%            | 12.3%            | 6.7%           | 6.3%   |
| related_drugs_mentioned | 24.0%            | 10.3%            | 4.3%           | 12.0%  |
| cml_diagnosed           | 43.7%            | 19.0%            | 7.0%           | 12.3%  |
| cml_in_regression       | 21.7%            | 8.7%             | 1.7%           | 3.7%   |
| aml_diagnosed           | 1.7%             | 8.0%             | 4.7%           | 6.0%   |
| blast_phase_cml         | 0.7%             | 1.3%             | 0.7%           | 1.0%   |
| bmt_history             | 14.3%            | 17.0%            | 8.0%           | 7.3%   |
| acute_phase_cml         | 1.0%             | 3.7%             | 0.3%           | 1.3%   |


# Finetuning
- To run or fine-tune the model, please navigate to the `Fine_tuning` directory in this GitHub repository.
- Before running please make sure the following dependancies are installed

| Package / Tool       | Recommended Version | Notes |
|---------------------|------------------|-------|
| Python              | 3.10+             | Tested with 3.10–3.11 |
| Jupyter Notebook    | 6.5+              | Required to run provided notebooks |
| pandas              | 2.0+              | For reading and manipulating Parquet files |
| numpy               | 1.24+             | Required for numerical operations |
| torch (PyTorch)     | 2.1+              | For model fine-tuning and inference |
| transformers (HF)   | 4.35+             | HuggingFace library for LLMs |
| accelerate          | 1.12+             | Optional but recommended for multi-GPU training |
| datasets            | 2.12+             | For handling large datasets if needed |
| tqdm                | 4.65+             | For progress bars during training |
| unsloth             | latest            | Required for DeepSeek / OpenHermes model usage |

- To reproduce these tables, please run content in `Concept_extraction_counts.ipynb` to grab counts for each concept for each LLM json file.
- Please run `Person_ID_generation.ipynb` to grab `Source_person_values` from LLM json files. You will need `latest_notes.parquet` to compare and match each note with its source clinical note.
- After running `Source_person_values` please navigate to [UCSF Emerse](https://emerse.ucsf.edu/) and enter in values for `Source_person_values` to get Demographics.


- Open the notebook `Fine_tuning(1).ipynb`. In this notebook, `openhermes-2.5-mistral-7b.Q2_K` was fine-tuned using 200 manually annotated notes. These notes can be found in the same directory under the name `manual_notes_200.json`.  
- Please note: to run fine-tuning, you will also need `latest_notes.parquet` in your directory, as it is used to extract the note text. The `manual_notes_200.json` file contains manually annotated versions of the first 200 notes from `latest_notes.parquet`. You can access them with:
  
```python
import pandas as pd
df = pd.read_parquet('latest_notes.parquet')
first_200 = df['note_text'].iloc[0:200].values
```
- Running `Fine_tuning(1).ipynb` will produce a `merged_model` in your directory.  
- After fine-tuning, run `Fine_tune_application.ipynb` to use your newly tuned model.  
- After LLM inference with `Fine_tune_application.ipynb`, you will need to verify your results. An additional 100 manually annotated notes are included in the directory for comparison. These notes correspond to note numbers 201–300 and are saved as `100_manual_notes_test.json`. Please use these notes to evaluate and compare your results.
- If you are not interested in a tuned OpenHermes model, the notebooks provided in this directory can run larger models with minimal modification.  
- Please note: fine-tuning larger models will likely require more computational resources.  

| Model | Recommended GPU | VRAM | Notes |
|-------|----------------|------|-------|
| openhermes-2.5-mistral-7b.Q2_K | NVIDIA A100 / RTX 4090 | 24–40 GB | Can fine-tune on moderate dataset (~200 notes) |
| Qwen-14B | NVIDIA A100 80GB | 80 GB | Requires high VRAM, consider gradient checkpointing |
| MEDGEMMA-4B | RTX 3090 / A100 40GB | 24–40 GB | Smaller model, less compute-intensive |
| Larger LLMs (e.g., 30B+) | Multiple A100s or equivalent | 80+ GB | Requires distributed training and high-end compute cluster |

# Fine-Tune openhermes-2.5-mistral-7b.Q2_K results
- Results of fine-tuning are saved to `parsed_outputs.json`
  
### Model vs Manual Annotation Accuracy

| Field                   | Baseline Accuracy (%) | Fine-Tuned Accuracy (%) | Improvement (%) |
|--------------------------|---------------------|------------------------|----------------|
| imatinib_mentioned       | 80.95               | 82.14                  | +1.19          |
| related_drugs_mentioned  | 72.62               | 77.38                  | +4.76          |
| cml_diagnosed            | 71.43               | 75.00                  | +3.57          |
| cml_in_regression        | 84.52               | 90.48                  | +5.96          |
| aml_diagnosed            | 84.52               | 90.48                  | +5.96          |
| blast_phase_cml          | 96.43               | 96.43                  | +0.00          |
| bmt_history              | 72.62               | 76.19                  | +3.57          |
| acute_phase_cml          | 94.05               | 96.43                  | +2.38          |


# Discussion
For this analysis, LLM inference was performed exclusively on progress notes. Other note types, such as structured drug prescription tables, medication administration records, or lab reports, were not included. As a result, the counts reflect mentions present in progress notes only and may not take into account related mentions in other databases.

1. Structured EHR Data

Structured EHR data using the filters assigned in Atlas captures more than just progress notes such as high-confidence prescription and diagnosis records. For example, imatinib_mentioned has 72 expected patients, while cml_diagnosed has 1936 expected patients. When filtering the clinical notes for only latest progress_notes the total amount of patients decrease to 1762 patients. This is likely because, these patients are new to the UCSF ecosystem or patients were only seen one time for a consult and decided to recieve care else where. Structured data is precise when present but may miss historical or context-specific mentions, such as prior therapy discussions or treatment changes not yet updated in the system.

2. Concept Extraction from Progress Notes

Regular expression was used to extract concepts from the fitlered clinical notes. From the regular expression more patients than expected were identified as taking imatinib.

LLM-based concept extraction identifies additional patients beyond structured EHRs. 

For instance:
MedGemma-4B flagged 316 patients with imatinib_mentioned, OpenHermes-2.5-7B flagged 328 patients, and DeepSeek-R1-14b flagged 191 patients.

Although only the most recent progress notes were analyzed, these extractions capture mentions that structured fields often miss—for example, documentation of therapy initiation, prior treatment history, or context-specific notes about patient response. Manual review of a subset confirmed that many of these extractions represent true mentions.

The higher number of mentions likely reflects an artifact not accounted for during cohort construction. While imatinib is primarily used to treat CML, it also has indications in other diseases such as gastrointestinal stromal tumors. Because progress notes capture a patient’s full treatment history at UCSF, some patients may have been prescribed imatinib for other conditions in the past, which contributes to the higher-than-expected counts.

3. Comparative Observations Across Models

DeepSeek-R1-14b is more conservative, producing fewer true positives compared to MedGemma and OpenHermes.

MedGemma-4B and OpenHermes-2.5-7B capture a larger fraction of potential mentions, indicating model sensitivity affects coverage.

Across all models, concept extraction improves sensitivity relative to structured EHR data.

4. Recommendation

Structured EHR data is still a reliable way to build cohorts and identify patients who are on the drug - LLM's are a valid approach to concept extraction for a drug of interest, however LLM's are dependant on a few key points. 
- LLM's are dependant on the type of notes that they are given.
- LLM's are technically expensive to setup.
- LLM's are costly to run and dependant on avaliable hardware.
Structured EHR data is reliable for confirmed prescriptions and formal diagnoses.

LLM extraction from progress notes enhances coverage and context.

A hybrid approach — combining structured EHR data with LLM-extracted concepts, validated with spot manual review — provides the most accurate view of drug usage.

5. Limitation and Future Directions

This analysis offers a single perspective, focusing solely on progress notes. Some notes contain placeholders such as “see scanned image for notes”, which may prevent extraction of relevant information. A more comprehensive evaluation would incorporate additional patient-associated data sources—such as prescription records, medication administration logs, and laboratory results—and apply LLMs across these datasets. Integrating multiple sources could improve coverage, capture historical and context-specific information, and provide a richer, more accurate understanding of patient treatment trajectories.

6. Fine-Tuned Model Discussion

After fine-tuning OpenHermes-2.5-7B, we observed an increase in accuracy across all label fields. However, these results should be interpreted cautiously. The model was fine-tuned using only 200 samples and evaluated on approximately 100 samples.

During evaluation, the model skipped two notes for reasons that are unclear, though this may have been due to a latency or processing issue during inference. Despite these limitations, fine-tuning improved performance, particularly for fields that were previously more challenging, such as drug mentions and clinical regression labels.

This improvement demonstrates that fine-tuning is a valid approach to increasing model accuracy. The main drawbacks are the need for manual annotation (or a larger model to assist with annotation) and the additional time and computational resources required to fine-tune and run the models. Additionally, because annotations are manually created, there is always a risk of mislabeling, which can affect model performance.

Nevertheless, when done carefully, fine-tuning can be a powerful tool for improving the accuracy of LLMs, particularly for extracting information from clinical notes.
























       (\_._/)  
       ( o o )  
      ==( V )==  
     /  |   |  \  
    (   |   |   ) 
     ^^     ^^
