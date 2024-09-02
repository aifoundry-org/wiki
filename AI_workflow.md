# This is how building of AI+CI/CD process could look like [WIP]

## Key ideas:

1. Continuous delivery when possible;  
2. Delivery when even when we don’t have enough info (metrics, requirements, etc);  
   1. Make progression on AI models  
3. Building a CI/CD process for AI as a result.  
   1. Ideal scenario: updating models on new training data and through releases 

## Key stages:

1. **Proof of concept: show we can gather the building blocks for an AI model and that it can yield some interesting results based on the premise**   
   We don’t know anything apart from basic product and system requirements. AI models:  selected from what’s existing in open access based on system requirements, available integrations, and general performance. Data: initial dataset collection is happening.  
   TODO on this stage:  
   1. Formulate initial system and product **requirements** for an AI system;  
   2. Collect initial **dataset**;  
   3. Identify and list potentially **suitable AI models**;  
   4. Perform **manual evaluation** using what was done on previous stages.

	  
	**Decision point: Is this a go or no-go for being a feasible product with AI?**   
	  
*Open question: What can be shared with Engineering at this point? What decisions can be made? What, if any, technical design choices, can be made at this point knowing what we know? What does Engineering need at this point in order not to be blocked?* 

Outputs**: TODO**

	

2. **Product Scoping for AI/Model choice evaluation: in this phase we continue exploratory work in order to make a solid recommendation for the product scope as it relates to the AI work**    
   General metrics/benchmarks are selected based on product requirements, product metrics are selected as well, initial dataset collection and labeling happens, initial results on the benchmarks are obtained, however, the amount of data is not enough to fine-tune and/or to train custom models. Now the choice of available models is metrics-based.   
   TODO on this stage:  
   1. Proceed with dataset collection, start labeling  
   2. Make a list of metrics/benchmarks somehow related to our product requirements   
   3. Create a benchmark runner with metrics selected from point 2\.  
   4. Make a more funded selection/create a set of initial metrics/comparison with other available tools if they exist

	  
Requirements: **TODO** (what would be blocking this work? What needs to happen? What needs to be defined?) 

**Decision point: Based on experimentation and specific requirements, we recommend choosing approach X for building out the product**. Note: further refinement will still be needed throughout development.  

*Open question: How much of this work should be completed before Engineering development starts, in theory? Which Engineering efforts should be on hold until this work is completed?* 

	Outputs: **TODO** 

3. **Model Fine Tuning: Improving upon the foundation model**    
   Enough data is collected to fine-tune the model and to launch a full CI/CD process with periodical updates of the model trained/fine-tuned on the new data and/or user feedback.

4. **Model Redesign: Building out own model**  
   We collected enough data to train a fully custom model if necessary, the CI/CD process from stage 3 remains intact excluding longer duration of training, necessity to stop on checkpoints etc etc.
