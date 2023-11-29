#!/usr/bin/env python
# coding: utf-8

# In[16]:


dict={
   "download hallticket": "https://vnrvjietexams.net/Eduprime3Exam/HallTicket",
    "provide link download hallticket" : "https://vnrvjietexams.net/Eduprime3Exam/HallTicket",
    "Hall Ticket link":"https://vnrvjietexams.net/Eduprime3Exam/HallTicket",
    "exam result":"https://vnrvjietexams.net/eduprime3exam/results",
    "check exam result":"https://vnrvjietexams.net/eduprime3exam/results",
    "examination result link":"https://vnrvjietexams.net/eduprime3exam/results",
    "help find previous year paper":"https://vnrvjietexams.net/QP.html",
    "find previous paper":"https://vnrvjietexams.net/QP.html",
    "previous year examination paper":"https://vnrvjietexams.net/QP.html",
    "examination payment link":"https://vnrvjietexams.net/Eduprime3Exam/ExamFee",
    "exam fee payment":"https://vnrvjietexams.net/Eduprime3Exam/ExamFee",
    "pay exam fee":"https://vnrvjietexams.net/Eduprime3Exam/ExamFee",
    "download hallticket":"https://vnrvjietexams.net/Eduprime3Exam/HallTicket",
    "exam hall ticket":"https://vnrvjietexams.net/Eduprime3Exam/HallTicket",
    "rc / rv payment":"https://vnrvjietexams.net/Eduprime3Exam/RCRV",
    "correction payment page":"https://vnrvjietexams.net/Eduprime3Exam/RCRV",
    "recorrection payment":"https://vnrvjietexams.net/Eduprime3Exam/RCRV",
    "evaluation payment":"https://vnrvjietexams.net/Eduprime3Exam/RCRV",
    "revaluation payment":"https://vnrvjietexams.net/Eduprime3Exam/RCRV",
    "recorrection- revaluation":"https://vnrvjietexams.net/Eduprime3Exam/RCRV",
    "late exam notification":"http://vnrvjiet.ac.in/vnrexams.php",
    "find exam timetable":"https://vnrvjietexams.net/QP.html",
    "contact exam branch":"https://vnrvjietexams.net/ExamBranch/contact.html",
    "check exam result":"https://vnrvjietexams.net/eduprime3exam/results",
    "academic calendar":"Can you be more specific?",
    "tech roll list" : "Which roll list are you lookinf for?",
    "pay condonation fee":"B.Tech condonation or M.Tech condonation?",
    "condonation form":"B.Tech condonation or M.Tech condonation?",
    "Syllabus book":"Can you be more specific?",
    "academic regulation":"Academeic regulations for B.Tech or M.Tech or BBA?",
    "contact academic query": "http://vnrvjiet.ac.in/contactus.php",
    "e library page":"https://vnrvjiet.new.knimbus.com/user#/home",
    "contact details":"Can you be more specific?"
}


# In[17]:


from transformers import BertTokenizer

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the data
tokenized_inputs = tokenizer(list(dict.keys()), return_tensors='pt', padding=True, truncation=True, max_length=512)


# In[19]:


tokenized_inputs.keys()


# In[41]:


len(tokenized_inputs['attention_mask'])


# In[28]:


import torch

# Convert labels to one-hot encoding
labels = torch.eye(len(dict)).unsqueeze(0)  # shape: (1, 5, 5)


# In[35]:


len(labels[0][0])


# In[42]:


from transformers import BertForSequenceClassification, AdamW


# In[43]:


# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(dict))


# In[44]:


# Define the optimizer and learning rate
optimizer = AdamW(model.parameters(), lr=5e-5)


# In[57]:


#checking the dimensions
print(tokenized_inputs['input_ids'])
print(tokenized_inputs['attention_mask'])
print(labels)


# In[58]:


# Fine-tune the model on the dataset
model.train()
#target = target.view(-1, 1)
loss = model(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], labels=labels).loss
#loss = criterion(outputs.view(-1, num_labels), labels.view(-1))
loss.backward()
optimizer.step()


# In[ ]:


#just some debuging ignore this
#3333333333333 Reshape the target tensor
#target = target.view(-1, 1)

# Compute the loss
#loss = criterion(output, target)


# In[ ]:


# Put the model into evaluation mode
model.eval()

# Tokenize the test data
test_inputs = tokenizer(["download hallticket"], return_tensors='pt', padding=True, truncation=True, max_length=512)

# Make predictions on the test data
with torch.no_grad():
    outputs = model(test_inputs['input_ids'], test_inputs['attention_mask'])
    predictions = torch.argmax(outputs[0], dim=1)

# Print the predicted label
predicted_label = list(dict.keys())[predictions[0].item()]
print("Predicted label:", predicted_label)

