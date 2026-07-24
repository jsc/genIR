This is the test collection to be used for the First Challenge in the GenIR Workshop in SIGIR 2026. The collection is a modified version of the first MSMARCO passage collection, so it should be easy to use for most IR researchers. The collection is composed of the following files:

1. genIR.collection.tsv - This is the augmented MSMARCO collection containing around 100,000 passages generated using LLMs for 1,796 queries, where each query will have multiple generated passages we have created using a variety of different techniques. You should be able to index this collection the same way you would index an MSMARCO collection.

2. queries.train.tsv - The training query set used for augmentation.

3. gen.train.tsv - The passage IDs for all of the generated passages, indexed by query id.

4. qrels.train.tsv - Relevance assessments derived from the original collation for the training queries, should you wish to do a statistical analysis on how generated passages might be affecting your ranker.

5. triples.train.tsv - A training file that contains 73,919 pairs of real vs generated passages indexed by query id. So, <QueryID, RealPsg, GenPsg>. You are free to use this to finetune a ranking or classifier model that can distinguish between LLM generated passages and passages extracted from real web documents. This should be enough samples to train a model for the challenge task.


6. queries.val.tsv - The validation set to be used for your experiments.

7. gen.val.tsv - The LLM generated passages indexed by query. You can use this to determine if your model is working.

8. qrels.val.tsv - As above, the relevance assessments derived from the original collection to measure ranker performance. 

9. queries.test.tsv - The test queries that you will use to submit your final run files.

The submission formation is straightforward -- Please submit a tab separated file containing the following:

QueryID <tab> Rank <tab> PassageID

You should return the top 200 documents for each query, with rank starting at 0. If you do not zero index your rank values, then then your submission will be underscored, i.e. perform worse than you would expect.

We hope you enjoy this increasingly important real world challenge!

- GenIR Workshop Organizing Team
