RAG application is build in Two steps.
1. Index process which reads the different PDF files (Document loader), Split in to Chunks (Text splitter), Apply EMbedding and store in Vector DB. 
2. Next process is streamlit and AI application which has a frontend and process which calls AI model use the FAISS Vector DB, get query from farmers and give the response. 
3. If updates are made in to PDFs, then index has to be run again. if not only the Agriapp program has to be run. 

Run this as 2 steps.
1. Index.py (To load the pdf in to FAISS as vectors)
2. Agriapp.py as streamlit run to load the UI , invoke LLM model 

If no change is done to PDFs, then run only the Agriapp.py . 
