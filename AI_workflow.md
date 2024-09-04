# This is how building of AI+CI/CD process could look like [WIP]

This wiki page is an attempt to outline the general strategy for developing an AI model for an AI-driven product, the key ideas we want to achieve during development, the assumptions we're working under, and the steps necessary to achieve them.

author: [@robolamp](https://github.com/robolamp)

## Key ideas:

1. Continuous delivery when possible;  
2. Delivery when even when we don’t have enough info (metrics, requirements, etc);  
   1. Make progression on AI models  
3. Building a CI/CD process for AI as a result.  
   1. Ideal scenario: updating models on new training data and through releases 

## Key assumptions:

1. The product requirements can be solved with AI  
2. The product requirements need AI to be solved efficiently  
3. There’s foundation models\* available in the required domain, and could work as a basis.  
   

\* foundation model — A large, general-purpose model that can be fine-tuned for specific applications, much like how a versatile tool can be used for different jobs with a few adjustments ([https://en.wikipedia.org/wiki/Foundation\_model](https://en.wikipedia.org/wiki/Foundation\_model))

## Key stages:

1. **Proof of concept: show we can gather the building blocks for an AI model and that it can yield some interesting results based on the premise**   
   We don’t know anything apart from basic product and system requirements. AI models:  selected from what’s existing in open access based on system requirements, available integrations, and general performance. Data: initial dataset collection is happening.

   <u>TODO on this stage</u>:
   * Formulate initial system and product **requirements** for an AI system;  
   * Collect initial **dataset**;  
   * Identify and list potentially **suitable AI models**;  
   * Perform **manual evaluation** using what was done on previous stages.

	<u>Outputs</u>: 
   * Product requirements;  
   * Initial dataset;  
   * Potentially suitable foundation models are selected.

2. **Product Scoping for AI/Model choice evaluation: in this phase we continue exploratory work in order to make a solid recommendation for the product scope as it relates to the AI work**    
   General metrics/benchmarks are selected based on product requirements, product metrics are selected as well, initial dataset collection and labeling happens, initial results on the benchmarks are obtained, however, the amount of data is not enough to fine-tune and/or to train custom models. Now the choice of available models is metrics-based.

   <u>TODO on this stage</u>:  
   * Proceed with dataset collection, start labeling
   * Make a list of metrics/benchmarks somehow related to our product requirements 
   * Create a benchmark runner with metrics selected from point 2.
   * Make a more funded selection/create a set of initial metrics/comparison with other available tools if they exist
   
	<u>Outputs</u>:
   * Dataset (and/or its labeled part) big enough to perform numerical evaluation;
   * Labeling (human feedback) pipeline;
   * Evaluation pipeline;
   * More funded fundamental model + hyperparameters choice selection.

3. **Model Fine Tuning: Improving upon the foundation model**    
   Enough data is collected to fine-tune the model and to launch a full CI/CD process with periodical updates of the model trained/fine-tuned on the new data and/or user feedback.
   <u>TODO on this stage:</u>
   * Continue with labeling/human feedback gathering/etc;
   * Perform fine-tuning methods evaluation; find the best working for the project;
   * Create a training pipeline for fine-tuning (and reuse eval pipeline from stage 2);
   * Create a procedure for delivery of fine-tuned models.

   <u>Outputs:</u>
   * Dataset (and/or its labeled part) big enough to LoRa or proper fine-tuning and evaluation of fine-tuned models evaluation;
   * Fine-tuning (short training) pipeline and all the required infrastructure;
   * Adapters for fine-tuned foundational models.


4. **Model Redesign: Building out own model**  
   We collected enough data to train a fully custom model if necessary, the CI/CD process from stage 3 remains intact excluding longer duration of training, necessity to stop on checkpoints etc etc.
