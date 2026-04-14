# Benchmark instructions
The goal of benchmark script is to evaluate memory solution. This document explains how to evaluate different kind of question types. 

## temp-reasoning
I will give you a question, a correct answer, and a response from a model. Please answer
yes if the response contains the correct answer. Otherwise, answer no. If the response is
equivalent to the correct answer or contains all the intermediate steps to get the correct
answer, you should also answer yes. If the response only contains a subset of the information
required by the answer, answer no. In addition, do not penalize off-by-one errors for the
number of days. If the question asks for the number of days/weeks/months, etc., and the
model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model’s
response is still correct.
## knowledge-update
I will give you a question, a correct answer, and a response from a model. Please answer
yes if the response contains the correct answer. Otherwise, answer no. If the response
contains some previous information along with an updated answer, the response should be
considered as correct as long as the updated answer is the required answer.
## single-session-preference
I will give you a question, a rubric for desired personalized response, and a response from a
model. Please answer yes if the response satisfies the desired response. Otherwise, answer
no. The model does not need to reflect all the points in the rubric. The response is correct
as long as it recalls and utilizes the user’s personal information correctly.
## Other question types
I will give you a question, a correct answer, and a response from a model. Please answer
yes if the response contains the correct answer. Otherwise, answer no. If the response
is equivalent to the correct answer or contains all the intermediate steps to get the correct
answer, you should also answer yes. If the response only contains a subset of the information
required by the answer, answer no.