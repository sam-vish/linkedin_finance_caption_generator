from llm import initialize_llm
from tools import get_tools
from embeddings import retrieve_relevant_examples

def generate_linkedin_post(finance_topic):
    """Generate a LinkedIn post on the given finance topic."""
    llm = initialize_llm()
    tools = get_tools()

    researcher = {
        "role": 'Financial Researcher',
        "goal": f'Research the given finance topic: "{finance_topic}".',
        "backstory": """You are a knowledgeable financial researcher with expertise in various financial topics.
        You excel at gathering accurate and comprehensive information on finance-related subjects, including specific markets and investment strategies.""",
        "tools": tools,
    }

    writer = {
        "role": 'LinkedIn Financial Influencer',
        "goal": f'Write a short, engaging, and easy-to-understand LinkedIn post on {finance_topic}.',
        "backstory": """You are a popular financial influencer on LinkedIn, known for explaining complex financial concepts in simple terms.
        Your posts are concise, use everyday language, and incorporate numbers and simple analogies to make financial topics accessible to everyone.""",
        "tools": [tool for tool in tools if tool.name == "Auto Response"],
    }

    research_task = f"""Conduct thorough research on the topic: "{finance_topic}".
    Focus on key aspects such as:
    1. Current market trends
    2. Investment strategies specific to this topic
    3. Potential risks and rewards
    4. Legal or regulatory considerations
    5. Expert opinions and case studies
    6. Relevant statistics and data

    Provide a comprehensive research report on the given finance topic, including actionable insights and specific data points."""

    # Step 1: Conduct research
    research_result = llm.predict(f"As a financial researcher, {research_task}")

    # Step 2: Generate LinkedIn post
    write_post_task = f"""Based on the research, create an engaging LinkedIn post about {finance_topic}.
    Your post MUST closely mimic the style, tone, and format of the following examples. Pay careful attention to these examples and ensure your post matches their format exactly:

    {retrieve_relevant_examples(finance_topic)}

    Your post MUST adhere to these strict guidelines:
    1. Start with a bold, attention-grabbing statement or question related to {finance_topic}.
    2. Use short, punchy paragraphs, often just one or two sentences each.
    3. Include 2-3 relevant statistics or numbers, presented in a memorable way (e.g., "125% returns").
    4. Incorporate bullet points or numbered lists for easy readability.
    5. Use some text formatting like **bold** or CAPS for emphasis (but don't overdo it).
    6. Include a simple, relatable analogy or real-life example to explain a complex concept.
    7. Provide one or two actionable tips or insights.
    8. End with a thought-provoking question or call-to-action.
    9. Keep the overall tone conversational and slightly provocative, as if you're challenging conventional wisdom.
    10. Aim for around 200-300 words total.
    11. DO NOT use any emojis. Keep it simple, formal, and clean.
    12. End with a question to encourage engagement.
    13. The language should be human-like, not AI-generated.

    Remember, the goal is to sound like a knowledgeable friend giving advice, not a textbook or formal report. Make it engaging, informative, and easy to read quickly on a LinkedIn feed.

    IMPORTANT: Do not deviate from this format. Your post should look and feel exactly like the examples provided."""

    linkedin_post = llm.predict(f"As a LinkedIn financial influencer, use this research to {write_post_task}\n\nResearch:\n{research_result}")

    return research_result, linkedin_post