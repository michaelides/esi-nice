1. Add button to sidebar to change the response style: concise and to the point, normal, explanatory and unnecessarily pedantic 
2. Add button to disable/enable/force tool selection for each of the tools
# 3. Download button for RAG pdfs
4. Report html link when finding rag files in web_markdown folder
5. Show citations of weblinks used. 
# 6. Fix page scrolling
# 7. Add icon to export to markdown or word document
# 8. Add clear history button
# 9. Add persistent long-term memory and user management via cookies. 
10. Ensure that all different uploaded file formats are correct.
# 11. Add support for SPSS files
# 12. Add button to request for a different response 🔄
13. In the chat box allow to navigate to previous questions in the prompt using the up-down arrow keys
14. Add bm25 retriever to create hybrid search. Hybrid Search (similarity_search + BM25Retriever) e.g. https://github.com/chroma-core/chroma/issues/1686
15. Data view via itable
# 16. Use react agent from langgraph
17. Try to get the agent to extract data from the search results
# 18. Add button to download chat transcript and option to download specific response
19. Add verification of DOI links
20. Fix hallucinations of paper - use SS, RAG and wiki info

21. Initial suggested prompts seem to be always the same 
# 22. Add copy button after each chat message
# 23. Add upload data option
# 24.  Greeding message seems to always be the same
25. Fixed initial warning about cache
26. Add iframe support for dataframes
27. Ensure that code executaion works
28. Add more libraries about code execution (pymc, pystan, brms, rpy, etc)


add the ability to upload files in the sidebar. Acceptable files include SPSS files (.sav), Rdata, rds, csv, xlsx, pdf, docx, and md. If it is a document then the file should be read and user can ask questions about the content. The files should NOT be added to the RAG - but should be an alternative temporary resource for that user.  If it is a dataset it should be imported as a pandas data.frame and the user can ask questions about thier variables, what analysis to perform etc. They can also ask the ai to perform data analysis procedures. If the later the ai should use the code execution functionality to write python code and execute it to run the analysis                                      
