from transformers import pipeline



INSTRUCTIONS = """
    Your task is to answer questions from the course participants
    based on the provided context.

    Use the context to find relevant information and provide accurate
    answers. If the answer is not found in the context,
    respond with "I don't know."
"""

PROMPT_TEMPLATE = """
    QUESTION: {question}

    CONTEXT:
    {context}
""".strip()


class RAGBase:

    def __init__(
        self,
        index,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        course="llm-zoomcamp",
        model="Qwen/Qwen3-4B-Instruct-2507"
    ):
        self.index = index
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model


    def search(self, query, num_results=5):
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        # context = self.build_context(search_results)
        return self.prompt_template.format(
            instructions=INSTRUCTIONS,
            question=query, 
            context=search_results
        ).strip()


    def llm(self, prompt):
        message_history = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        generator = pipeline(
            "text-generation",
            model=self.model
        )

        result = generator(
            message_history,
            max_new_tokens=100,
            num_return_sequences=1
        )

        answer = result[0]["generated_text"]

        return answer[2]["content"]



    def rag(self, query):
        search_results = self.search(query)
        context = self.build_context(search_results)
        prompt = self.build_prompt(query, context)
        answer = self.llm(prompt)

        return answer

