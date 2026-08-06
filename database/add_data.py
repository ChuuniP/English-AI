"""
Script to manually add data into the writing_prompts table.
Run this once to seed the writing_prompts table with sample essay questions.
"""

from database_manager import DatabaseManager


def seed_writing_prompts(db: DatabaseManager) -> int:
    """Insert a fixed set of writing prompts. Returns the number of rows added."""

    prompts = [
        dict(
            task_type="opinion",
            topic_category="education",
            difficulty="Intermediate",
            question_text="Some people think that university students should study whatever they like. Others believe they should only be allowed to study subjects that will be useful in the future, such as those related to science and technology. Discuss both views and give your opinion.",
            min_words=250,
            suggested_time_minutes=40,
            tags="education;university;career"
        ),
        dict(
            task_type="discussion",
            topic_category="technology",
            difficulty="Advanced",
            question_text="In many countries, technology is increasingly used in classrooms to replace traditional teaching methods. To what extent do you agree or disagree with this trend?",
            min_words=250,
            suggested_time_minutes=40,
            tags="technology;education"
        ),
        dict(
            task_type="problem_solution",
            topic_category="environment",
            difficulty="Advanced",
            question_text="Air pollution in many cities has become a serious problem, especially for people's health. What are the causes of this problem, and what measures can be taken to solve it?",
            min_words=250,
            suggested_time_minutes=40,
            tags="environment;health;pollution"
        ),
        dict(
            task_type="advantage_disadvantage",
            topic_category="society",
            difficulty="Intermediate",
            question_text="More and more people are choosing to work from home instead of going to an office. What are the advantages and disadvantages of this trend?",
            min_words=250,
            suggested_time_minutes=40,
            tags="work;society;remote work"
        ),
        dict(
            task_type="opinion",
            topic_category="government",
            difficulty="Advanced",
            question_text="Some people believe that the government should be responsible for taking care of elderly citizens, while others think this responsibility should belong to individual families. Discuss both views and give your own opinion.",
            min_words=250,
            suggested_time_minutes=40,
            tags="government;family;elderly care"
        ),
        dict(
            task_type="discussion",
            topic_category="media",
            difficulty="Intermediate",
            question_text="Nowadays, many people get their news from social media rather than traditional newspapers or television. Is this a positive or negative development?",
            min_words=250,
            suggested_time_minutes=40,
            tags="media;news;social media"
        ),
        dict(
            task_type="problem_solution",
            topic_category="health",
            difficulty="Advanced",
            question_text="In many countries, obesity rates among children are rising rapidly. What are the causes of this issue, and what can be done to address it?",
            min_words=250,
            suggested_time_minutes=40,
            tags="health;children;obesity"
        ),
        dict(
            task_type="advantage_disadvantage",
            topic_category="economy",
            difficulty="Intermediate",
            question_text="Some countries encourage foreign companies to open branches within their borders. What are the advantages and disadvantages of this for the local economy?",
            min_words=250,
            suggested_time_minutes=40,
            tags="economy;globalization;business"
        ),
        dict(
            task_type="opinion",
            topic_category="culture",
            difficulty="Intermediate",
            question_text="Traditional customs and festivals are gradually disappearing in many societies due to globalization. Do you think this is a positive or negative trend?",
            min_words=250,
            suggested_time_minutes=40,
            tags="culture;globalization;tradition"
        ),
        dict(
            task_type="discussion",
            topic_category="education",
            difficulty="Advanced",
            question_text="Some people believe that examinations are the best way to assess a student's ability, while others think there are better methods of assessment. Discuss both views and give your opinion.",
            min_words=250,
            suggested_time_minutes=40,
            tags="education;exams;assessment"
        ),
        dict(
            task_type="opinion",
            topic_category="education",
            difficulty="Beginner",
            question_text="Do you agree or disagree with the following statement? Students learn more effectively by studying alone than by studying in groups. Use specific reasons and examples to support your answer.",
            min_words=300,
            suggested_time_minutes=30,
            tags="education;learning style"
        ),
        dict(
            task_type="opinion",
            topic_category="society",
            difficulty="Beginner",
            question_text="Some people prefer to live in a small town, while others prefer to live in a big city. Which do you prefer and why? Use specific reasons and details to support your answer.",
            min_words=300,
            suggested_time_minutes=30,
            tags="society;lifestyle;city"
        ),
        dict(
            task_type="opinion",
            topic_category="technology",
            difficulty="Intermediate",
            question_text="Do you agree or disagree with the following statement? Technology has made people more isolated from one another rather than more connected. Use specific reasons and examples.",
            min_words=300,
            suggested_time_minutes=30,
            tags="technology;relationships;isolation"
        ),
        dict(
            task_type="opinion",
            topic_category="work",
            difficulty="Beginner",
            question_text="Some people believe that it is important to have a job they enjoy, even if it pays less. Others think that earning a high salary is more important than job satisfaction. Which view do you agree with?",
            min_words=300,
            suggested_time_minutes=30,
            tags="work;career;salary"
        ),
        dict(
            task_type="opinion",
            topic_category="environment",
            difficulty="Intermediate",
            question_text="Do you agree or disagree with the following statement? Individuals can do little to reduce environmental problems; only governments and large businesses can make a real difference.",
            min_words=300,
            suggested_time_minutes=30,
            tags="environment;responsibility;government"
        ),
        dict(
            task_type="direct_question",
            topic_category="education",
            difficulty="Advanced",
            question_text="Some parents believe that children should spend most of their free time studying, while others believe children should be free to play and pursue hobbies. Discuss both views and give your opinion.",
            min_words=250,
            suggested_time_minutes=40,
            tags="education;childhood;parenting"
        ),
        dict(
            task_type="problem_solution",
            topic_category="society",
            difficulty="Intermediate",
            question_text="In many big cities, traffic congestion has become a major problem. What are the main causes of this issue, and what solutions can be implemented?",
            min_words=250,
            suggested_time_minutes=40,
            tags="transportation;city;traffic"
        ),
        dict(
            task_type="opinion",
            topic_category="technology",
            difficulty="Advanced",
            question_text="Some people think that the increasing use of computers and the internet has had a huge impact on children's ability to think and study creatively. To what extent do you agree or disagree?",
            min_words=250,
            suggested_time_minutes=40,
            tags="technology;children;creativity"
        ),
        dict(
            task_type="discussion",
            topic_category="government",
            difficulty="Intermediate",
            question_text="Some people think that the government should provide free healthcare for all citizens, while others believe individuals should pay for their own medical care. Discuss both views and give your opinion.",
            min_words=250,
            suggested_time_minutes=40,
            tags="healthcare;government;economy"
        ),
        dict(
            task_type="advantage_disadvantage",
            topic_category="education",
            difficulty="Intermediate",
            question_text="An increasing number of students are choosing to study abroad for their higher education. What are the advantages and disadvantages of this trend?",
            min_words=250,
            suggested_time_minutes=40,
            tags="education;study abroad;culture"
        ),
        dict(
            task_type="opinion",
            topic_category="environment",
            difficulty="Advanced",
            question_text="Some people believe that the best way to protect the environment is to increase the price of fuel, while others think there are better solutions. Discuss both views and give your opinion.",
            min_words=250,
            suggested_time_minutes=40,
            tags="environment;fuel;policy"
        ),
        dict(
            task_type="problem_solution",
            topic_category="work",
            difficulty="Intermediate",
            question_text="Many employees today report high levels of stress in the workplace. What are the causes of workplace stress, and what can employers do to reduce it?",
            min_words=250,
            suggested_time_minutes=40,
            tags="work;stress;mental health"
        ),
        dict(
            task_type="discussion",
            topic_category="media",
            difficulty="Intermediate",
            question_text="Some people think that celebrities such as film stars and athletes should use their fame to help raise awareness of important social issues. Others think this is not their responsibility. Discuss both views and give your opinion.",
            min_words=250,
            suggested_time_minutes=40,
            tags="media;celebrity;social responsibility"
        ),
        dict(
            task_type="advantage_disadvantage",
            topic_category="technology",
            difficulty="Intermediate",
            question_text="Online shopping is becoming increasingly popular around the world. What are the advantages and disadvantages of this development for both consumers and businesses?",
            min_words=250,
            suggested_time_minutes=40,
            tags="technology;shopping;economy"
        ),
        dict(
            task_type="opinion",
            topic_category="society",
            difficulty="Beginner",
            question_text="Some people believe that living in a large family with many relatives is beneficial, while others think it is better to live in a small nuclear family. Discuss both views and give your own opinion.",
            min_words=250,
            suggested_time_minutes=40,
            tags="family;society;lifestyle"
        ),
        dict(
            task_type="opinion",
            topic_category="education",
            difficulty="Beginner",
            question_text="Some people believe that university students should be required to attend classes, while others believe attendance should be optional. Which position do you agree with?",
            min_words=300,
            suggested_time_minutes=30,
            tags="education;university;attendance"
        ),
        dict(
            task_type="opinion",
            topic_category="health",
            difficulty="Beginner",
            question_text="Do you agree or disagree with the following statement? People should exercise regularly to stay healthy, even if it means giving up leisure activities they enjoy.",
            min_words=300,
            suggested_time_minutes=30,
            tags="health;exercise;lifestyle"
        ),
        dict(
            task_type="opinion",
            topic_category="environment",
            difficulty="Intermediate",
            question_text="Some people think that zoos are cruel and should be closed, while others believe zoos play an important role in conservation and education. Which opinion do you agree with?",
            min_words=300,
            suggested_time_minutes=30,
            tags="environment;animals;conservation"
        ),
        dict(
            task_type="opinion",
            topic_category="society",
            difficulty="Beginner",
            question_text="Do you agree or disagree with the following statement? It is more important for children to learn practical skills than academic subjects.",
            min_words=300,
            suggested_time_minutes=30,
            tags="education;children;skills"
        ),
        dict(
            task_type="opinion",
            topic_category="technology",
            difficulty="Intermediate",
            question_text="Some people believe that artificial intelligence will create more job opportunities than it destroys. Do you agree or disagree with this statement?",
            min_words=300,
            suggested_time_minutes=30,
            tags="technology;AI;employment"
        ),
        dict(
            task_type="problem_solution",
            topic_category="health",
            difficulty="Advanced",
            question_text="Mental health problems among young people are increasing in many countries. What are the reasons for this, and what steps can be taken to address the issue?",
            min_words=250,
            suggested_time_minutes=40,
            tags="health;mental health;youth"
        ),
        dict(
            task_type="opinion",
            topic_category="government",
            difficulty="Intermediate",
            question_text="Some people think that the government should invest more money in public transportation rather than building new roads. To what extent do you agree or disagree?",
            min_words=250,
            suggested_time_minutes=40,
            tags="government;transportation;investment"
        ),
        dict(
            task_type="discussion",
            topic_category="culture",
            difficulty="Intermediate",
            question_text="Some people believe that it is important for children to learn a foreign language from a young age, while others think it is not necessary until they are older. Discuss both views and give your opinion.",
            min_words=250,
            suggested_time_minutes=40,
            tags="education;language;childhood"
        ),
        dict(
            task_type="advantage_disadvantage",
            topic_category="work",
            difficulty="Intermediate",
            question_text="More companies are allowing their employees to choose their own working hours instead of a fixed schedule. What are the advantages and disadvantages of this practice?",
            min_words=250,
            suggested_time_minutes=40,
            tags="work;flexibility;productivity"
        ),
        dict(
            task_type="opinion",
            topic_category="economy",
            difficulty="Advanced",
            question_text="Some people believe that unpaid community service should be a compulsory part of high school education. To what extent do you agree or disagree?",
            min_words=250,
            suggested_time_minutes=40,
            tags="education;community service;society"
        ),
        dict(
            task_type="opinion",
            topic_category="work",
            difficulty="Beginner",
            question_text="Some people prefer to work for a large company, while others prefer to work for a small company. Which do you think is better, and why?",
            min_words=300,
            suggested_time_minutes=30,
            tags="work;career;company size"
        ),
        dict(
            task_type="opinion",
            topic_category="culture",
            difficulty="Beginner",
            question_text="Do you agree or disagree with the following statement? Learning about other cultures is essential for success in today's world.",
            min_words=300,
            suggested_time_minutes=30,
            tags="culture;globalization;education"
        ),
        dict(
            task_type="opinion",
            topic_category="society",
            difficulty="Intermediate",
            question_text="Some people think that success is mainly determined by hard work, while others believe it depends more on luck and circumstances. Which view do you agree with?",
            min_words=300,
            suggested_time_minutes=30,
            tags="success;society;philosophy"
        ),
        dict(
            task_type="problem_solution",
            topic_category="environment",
            difficulty="Intermediate",
            question_text="Plastic waste has become a serious environmental problem in many parts of the world. What are the main causes of this problem, and what can be done to reduce it?",
            min_words=250,
            suggested_time_minutes=40,
            tags="environment;plastic;waste"
        ),
        dict(
            task_type="opinion",
            topic_category="technology",
            difficulty="Beginner",
            question_text="Some people think that social media has a negative effect on how teenagers interact with each other in real life. To what extent do you agree or disagree?",
            min_words=250,
            suggested_time_minutes=40,
            tags="technology;teenagers;social media"
        ),
    ]

    added = 0
    for p in prompts:
        db.add_writing_prompt(**p)
        added += 1
    return added


if __name__ == "__main__":
    db = DatabaseManager()
    total_added = seed_writing_prompts(db)
    print(f"Added {total_added} writing prompts.")
    print(f"Total writing prompts in database: {len(db.list_all_writing_prompts())}")
    db.close()