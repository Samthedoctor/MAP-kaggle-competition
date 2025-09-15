# 📈 Charting Student Math Misunderstandings: A QLoRA Fine-Tuning Approach

This repository contains a solution for the [MAP - Charting Student Math Misunderstandings](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings) Kaggle competition. The goal is to develop a model that can predict a student's mathematical misconceptions from their written explanations.

## 🎯 Approach Overview

This solution frames the problem as a **multi-class classification task**. A specialized language model is fine-tuned to predict a combined label representing both the answer's correctness and the specific misconception, if any.

The core of this approach involves:
1.  **Problem Formulation**: Treating each unique `Category:Misconception` pair as a distinct class.
2.  **Prompt Engineering**: Structuring the input data (`Question`, `Student Answer`, `Student Explanation`) into a clear, formatted prompt for the language model.
3.  **Efficient Fine-Tuning**: Using **QLoRA** (Quantized Low-Rank Adaptation) to fine-tune a powerful, specialized math model on consumer-grade hardware.
4.  **Handling Class Imbalance**: Implementing a **custom trainer with a weighted loss function** to address the highly imbalanced nature of the dataset.
5.  **Custom Evaluation**: Building a metric function to evaluate the model using the competition's official metric, **Mean Average Precision @ 3 (MAP@3)**.

---

## 🛠️ Detailed Workflow

### 1. Data Preprocessing & Feature Engineering

The first step is to transform the raw data into a format suitable for a sequence classification model.

* **Target Column Creation**: The `Category` and `Misconception` columns were combined into a single target column named `category_misconception`. This turns the multi-step problem into a single multi-class classification task.
    ```python
    # Example: 'True_Correct' + ':' + 'NA' -> 'True_Correct:NA'
    df_train['category_misconception'] = df_train['Category'] + ':' + df_train['Misconception']
    ```

* **Prompt Engineering**: A structured prompt was created to provide the model with clear context for each data point. This helps the model understand the distinct parts of the input.
    ```python
    prompt_template = """Analyze the student's reasoning for the following math problem.

    ### Question:
    {QuestionText}

    ### Student's Answer Choice:
    {MC_Answer}

    ### Student's Explanation:
    {StudentExplanation}

    ### Analysis:"""
    df_train['prompt'] = df_train.apply(lambda row: prompt_template.format(...), axis=1)
    ```

* **Label Encoding**: The text-based `category_misconception` labels were converted into numerical IDs using `sklearn.preprocessing.LabelEncoder`. The mappings (`id_to_label` and `label_to_id`) were saved for inference.

* **Stratified Data Splitting**: To ensure the validation set is representative of the training data, a stratified split was performed. A custom logic was implemented to handle classes with only one member, which would otherwise cause an error. These single-member classes were moved directly into the training set.

### 2. Model & Fine-Tuning Strategy

To maximize performance while managing computational resources, a Parameter-Efficient Fine-Tuning (PEFT) strategy was employed.

* **Base Model**: **`Qwen/Qwen2.5-Math-1.5B`** was chosen as the base model. This is a powerful model specifically pre-trained for mathematical reasoning, making it an excellent candidate for this task. The model was loaded with a sequence classification head (`AutoModelForSequenceClassification`).

* **Quantization (QLoRA)**: To make fine-tuning feasible, the model was loaded in 4-bit precision using QLoRA.
    * **Quantization Type**: `nf4` (4-bit NormalFloat)
    * **Compute Datatype**: `torch.float16`
    ```python
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    ```

* **Low-Rank Adaptation (LoRA)**: LoRA was configured to add trainable, low-rank matrices to the model's attention layers. This drastically reduces the number of trainable parameters, making fine-tuning faster and more memory-efficient.
    * **Task Type**: `TaskType.SEQ_CLS` (Crucial for a classification task)
    * **LoRA Rank (`r`)**: 16
    * **LoRA Alpha (`lora_alpha`)**: 32
    * **Target Modules**: `["q_proj", "k_proj", "v_proj", "o_proj"]`
    
The final trainable parameters were only **0.67%** of the total model parameters.

### 3. Training Pipeline

The `transformers` `Trainer` API was used to orchestrate the training process.

* **Handling Class Imbalance**: The dataset is highly imbalanced, with many misconception classes appearing infrequently. To prevent the model from ignoring these rare classes, a **custom trainer with a weighted loss function** was created.
    1.  Class weights were calculated using `sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', ...)`
    2.  A `CustomTrainer` class inherited from the standard `Trainer`.
    3.  The `compute_loss` method was overridden to use `torch.nn.CrossEntropyLoss` with the pre-calculated weights.
    ```python
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor) # Using weights
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    ```

* **Evaluation Metric (MAP@3)**: A `compute_metrics` function was implemented to calculate the competition's primary evaluation metric, Mean Average Precision @ 3. This function takes the model's logits, identifies the top 3 predicted classes, and calculates the MAP@3 score against the true labels. This metric was used to identify the best-performing model checkpoint during training.

* **Training Arguments**: Key hyperparameters used for the training process include:
    * **Epochs**: 3
    * **Batch Size**: 2 (with `gradient_accumulation_steps=4`, for an effective batch size of 8)
    * **Optimizer**: `paged_adamw_8bit` (memory-efficient)
    * **Learning Rate**: `2e-4` with a `cosine` scheduler
    * **Evaluation Strategy**: `steps` (evaluate every 100 steps)

---

## 🚀 Conclusion

This approach provides a robust and efficient solution for classifying student math misconceptions. By combining a specialized math model with QLoRA, handling class imbalance through a weighted loss, and aligning the training process with the competition's MAP@3 metric, the model is well-equipped to accurately identify a wide range of misconceptions from student explanations.
