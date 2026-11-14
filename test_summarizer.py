import json
from agents.summarizer_agent import summarizer_node
# Sample output from Member 1's Research Agent
test_state = {
   "topic": "Latest developments in Generative AI",
   "research_results": [
       {
           "title": "OpenAI AI developments",
           "url": "https://openai.com/",
           "content": "OpenAI develops advanced AI models with improved reasoning and multimodal capabilities."
       },
       {
           "title": "Google Gemini developments",
           "url": "https://blog.google/",
           "content": "Google continues expanding Gemini capabilities across its products."
       },
       {
           "title": "Microsoft AI developments",
           "url": "https://www.microsoft.com/",
           "content": "Microsoft integrates generative AI into productivity and cloud services."
       }
   ],
   "errors": []
}
# Execute Summarizer Agent
result = summarizer_node(test_state)
print("\n========== SUMMARIZER OUTPUT ==========")
print(json.dumps(result, indent=4))
print("\n========== VALIDATION ==========")
if len(result.get("summary_points", [])) == 5:
   print("SUCCESS: Exactly 5 summary points generated!")
else:
   print("FAILED: Summary does not contain exactly 5 points.")
print("Sources:", result.get("sources", []))
print("Errors:", result.get("errors", []))