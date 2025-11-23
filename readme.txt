1. Install python 3.13.5 in custom path (all python packages downloaded to sitepackages and require more space )
2. Upgrade pip
python.exe -m pip install --upgrade pip
3. Generate contract docs using prompt in AI tool
4. pip install docx2pdf to generate contract from text or doc to pdf
5. Fix TqdmWarning - 
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade ipywidgets jupyterlab widgetsnbextension
6. pip install docx2pdf
7. Run script to check pdf creation
8. Then created llm python project with required UI and flow using,
 "google/flan-t5-base" as LLM Model (to match my system with 4 CPU, 16GB Ram)
 "sentence-transformers/all-MiniLM-L6-v2" as embedding model
 "chromaDB" as vector db
9. pip install -r requirements.txt
10. streamlit run app.py (default port 8501 - http://localhost:8501/) or streamlit run app.py --server.port=9900
11. To upgrade from google/flan-t5-base, run pip install -U langchain-huggingface
12. Changed to llm model google/gemma-2b-it & embedding model "sentence-transformers/all-MiniLM-L6-v2"
13. before that create account in huggingface (karthi4elite/123@Sastra)
14. access https://huggingface.co/google/gemma-2b-it
15. accept license and repo. Get approved by community
16. Generate token https://huggingface.co/settings/tokens
17. Type - read, name - gemma-access-token, token - hf_<token>
18. huggingface-cli login && Enter Token && Add token as git credential? (Y/n) n 
    && current active token is: `gemma-access-token` && huggingface-cli whoami
19. In powershell, setx HUGGINGFACEHUB_API_TOKEN "<input your hf token>"
20. add token var in code and then use token in tokenizer (chatbot & clause extractor)
21. first tensor model will take time - 5GB data downloaded
22. dedupe where applicable (ex: risk scorer, retriever, chatbot, app etc)
22. Updated risk_db data for more clause validation and risk analysis
23. Delete CHROMA_PERSIST_DIR = "chroma_store" path & start the app
24. implemented smart summary on top of basic summary
25. (optional)For unprecedented DB related error during chat, 
    downgrade from chromadb>=0.5.0 to chromadb==0.4.24 along with python version downgrade to 3.11
    and pip install "numpy<2.0.0" and turn chromadb telemetry off in risk_db and retriever. Then verify pip list
26. Finally tried with Python 3.12 and latest langchain and chromaDB
27.	Sample chat queries
    can we sign this contract?
    can you generate an image?
    can you notice any risk in contract?
28. Install “Microsoft C++ Build Tools”, Download Build Tools for Visual Studio.
    https://visualstudio.microsoft.com/visual-cpp-build-tools/
    Choose the workload: “Desktop development with C++”
    Make sure MSVC v14.x and Windows 10/11 SDK are checked.
    After installation, restart your terminal.    