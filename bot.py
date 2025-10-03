import discord
from discord.ext import commands
import os
import sqlite3
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

TOKEN = os.getenv('TOKEN')
AI_TOKEN = os.getenv('AI')

LEARNING_CHANNEL_ID = 1411335494234669076
TEACHER_ROLE_ID = 1418434355650625676
HELPER_ROLE_ID = 1352853011424219158
JUNIOR_HELPER_ROLE_ID = 1372300233240739920
STUDENT_ROLE_ID = 1341949236471926805

CONFIDENCE_THRESHOLD = 0.65
EASTER_EGG_CHANCE = 0.005
OWNER_ID = 1334138321412296725

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)

genai.configure(api_key=AI_TOKEN)
model = genai.GenerativeModel('gemini-pro')

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

conn = sqlite3.connect('bloom_knowledge.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS knowledge_base (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        embedding BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(question, answer)
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value REAL NOT NULL
    )
''')

cursor.execute('INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)', ('easter_egg_chance', 0.005))
conn.commit()

def get_easter_egg_chance():
    cursor.execute('SELECT value FROM settings WHERE key = ?', ('easter_egg_chance',))
    result = cursor.fetchone()
    return result[0] if result else 0.005

def set_easter_egg_chance(chance):
    cursor.execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)', ('easter_egg_chance', chance))
    conn.commit()

def get_embedding(text):
    return embedding_model.encode(text)

def find_similar_question(question_text):
    question_embedding = get_embedding(question_text)

    cursor.execute('SELECT id, question, answer, embedding FROM knowledge_base')
    rows = cursor.fetchall()

    if not rows:
        return None, 0.0

    best_match = None
    best_similarity = 0.0

    for row in rows:
        stored_embedding = np.frombuffer(row[3], dtype=np.float32)
        similarity = cosine_similarity(
            question_embedding.reshape(1, -1),
            stored_embedding.reshape(1, -1)
        )[0][0]

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = row

    if best_similarity >= CONFIDENCE_THRESHOLD:
        return best_match, best_similarity

    return None, best_similarity

def save_qa_pair(question, answer):
    try:
        embedding = get_embedding(question)
        embedding_bytes = embedding.astype(np.float32).tobytes()

        cursor.execute(
            'INSERT OR IGNORE INTO knowledge_base (question, answer, embedding) VALUES (?, ?, ?)',
            (question, answer, embedding_bytes)
        )
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"Error saving Q&A pair: {e}")
        return False

def is_helper(member):
    if not member:
        return False
    role_ids = [role.id for role in member.roles]
    return any(role_id in role_ids for role_id in [TEACHER_ROLE_ID, HELPER_ROLE_ID, JUNIOR_HELPER_ROLE_ID])

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    print(f'Bloom Learning System is active')
    try:
        synced = await bot.tree.sync()
        print(f'Synced {len(synced)} slash commands')
    except Exception as e:
        print(f'Failed to sync commands: {e}')

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    await bot.process_commands(message)

    if message.channel.id != LEARNING_CHANNEL_ID:
        return

    if message.reference and message.reference.resolved:
        if is_helper(message.author):
            try:
                replied_message = message.reference.resolved

                if replied_message.author.bot:
                    return

                question = replied_message.content.strip()
                answer = message.content.strip()

                if not question or not answer:
                    return

                saved = save_qa_pair(question, answer)

                if saved:
                    await message.add_reaction('✅')
                    print(f"Learned new Q&A: {question[:50]}... -> {answer[:50]}...")
            except Exception as e:
                print(f"Error processing helper reply: {e}")

    else:
        if not is_helper(message.author):
            try:
                question = message.content.strip()

                if not question:
                    return

                if random.random() < get_easter_egg_chance():
                    try:
                        prompt = f"""
You are Bloom, a Discord bot with a 0.5% chance of speaking.
When triggered, you must deliver the harshest roast possible.

Context: "{question}"

⚡ Special Rules:
- This is a rare event, so the roast must feel legendary.
- Absolutely brutal, clever, and unforgettable.
- Tie the insult to the context if possible.
- Keep it 1–2 sentences, max 300 characters.
- Must make the target regret ever triggering the 0.5% chance.
- Style: savage Discord roast × boss fight final attack.
- End with the deadliest emoji you can pick (💀, 🪦, ☠️, 🔥).

🔥 Legendary Roast Examples:
- "Congrats, you just unlocked Bloom's 0.5% roast… too bad your life stats are still stuck at tutorial level 💀"
- "Wow, you hit the 0.5% chance… the same odds as someone actually respecting you 🪦"
- "Lucky pull, unlucky life. Hitting this chance is the closest you'll ever get to winning anything ☠️"
- "Bloom speaks once in a thousand tries — and still finds you pathetic 🔥"

Now craft the harshest, rare-event roast ever for the given context.
"""
                        response = model.generate_content(prompt)
                        roast = response.text.strip()
                        await message.reply(roast)
                        print(f"0.5% roast triggered for: {message.author.name}")
                        return
                    except Exception as e:
                        print(f"Error generating roast: {e}")

                match, similarity = find_similar_question(question)

                if match:
                    stored_answer = match[2]
                    await message.reply(stored_answer)
                    print(f"Replied with stored answer (confidence: {similarity:.2%})")
                else:
                    print(f"No confident match found (best: {similarity:.2%}). Staying silent.")
            except Exception as e:
                print(f"Error processing student question: {e}")

@bot.tree.command(name='bloom_stats', description='View Bloom knowledge base statistics')
async def bloom_stats(interaction: discord.Interaction):
    if interaction.user.id != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    if interaction.channel_id != LEARNING_CHANNEL_ID:
        await interaction.response.send_message("This command only works in the learning channel.", ephemeral=True)
        return

    cursor.execute('SELECT COUNT(*) FROM knowledge_base')
    count = cursor.fetchone()[0]

    embed = discord.Embed(
        title="🌱 Bloom Knowledge Stats",
        description=f"I currently know **{count}** Q&A pairs!",
        color=discord.Color.green()
    )
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name='bloom_search', description='Search for similar questions in knowledge base')
async def bloom_search(interaction: discord.Interaction, query: str):
    if interaction.user.id != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    if interaction.channel_id != LEARNING_CHANNEL_ID:
        await interaction.response.send_message("This command only works in the learning channel.", ephemeral=True)
        return

    match, similarity = find_similar_question(query)

    if match:
        embed = discord.Embed(
            title="🔍 Found Similar Question",
            description=f"**Confidence:** {similarity:.2%}\n\n**Question:**\n{match[1]}\n\n**Answer:**\n{match[2]}",
            color=discord.Color.blue()
        )
        await interaction.response.send_message(embed=embed)
    else:
        await interaction.response.send_message(f"No match found above {CONFIDENCE_THRESHOLD:.0%} threshold. (Best: {similarity:.2%})")

@bot.tree.command(name='tellmeajoke', description='Bloom roasts someone')
async def tellmeajoke(interaction: discord.Interaction, user: discord.User = None):
    if interaction.user.id != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    target = user if user else interaction.user

    context = f"Roast this user: {target.name}"

    try:
        prompt = f"""
You are Bloom, a savage roasting bot.
Generate an absolutely brutal roast for this user.

Target: {target.name}

Rules:
- Harsh, clever, and unforgettable
- 1-2 sentences max
- Maximum 300 characters
- End with a deadly emoji (💀, 🪦, ☠️, 🔥)
- Make it legendary

Craft the most devastating roast possible.
"""
        response = model.generate_content(prompt)
        roast = response.text.strip()

        await interaction.response.send_message(f"{target.mention} {roast}")
    except Exception as e:
        await interaction.response.send_message("Failed to generate roast.", ephemeral=True)
        print(f"Error in tellmeajoke: {e}")

@bot.tree.command(name='say', description='Make Bloom say something')
async def say(interaction: discord.Interaction, message: str):
    if interaction.user.id != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    await interaction.response.send_message("Processing...", ephemeral=True)

    try:
        prompt = f"""
You are Bloom, a Discord bot assistant.
A user wants you to say something. Respond naturally based on their request.

User's message: "{message}"

Rules:
- Be helpful and friendly
- Keep it concise (under 500 characters)
- If they ask you to say something specific, say it naturally
- Add personality but stay on topic
"""
        response = model.generate_content(prompt)
        bloom_response = response.text.strip()

        await interaction.channel.send(bloom_response)
    except Exception as e:
        await interaction.followup.send("Failed to process message.", ephemeral=True)
        print(f"Error in say command: {e}")

@bot.tree.command(name='easter', description='Set the easter egg roast chance percentage')
async def easter(interaction: discord.Interaction, percentage: float):
    if interaction.user.id != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    if percentage < 0 or percentage > 100:
        await interaction.response.send_message("Percentage must be between 0 and 100.", ephemeral=True)
        return

    chance = percentage / 100.0
    set_easter_egg_chance(chance)

    await interaction.response.send_message(
        f"Easter egg roast chance set to **{percentage}%** ({chance:.4f})",
        ephemeral=True
    )
    print(f"Easter egg chance updated to {percentage}% by {interaction.user.name}")

@bot.tree.command(name='askbloom', description='Ask Bloom anything with critical thinking')
async def askbloom(interaction: discord.Interaction, question: str):
    if interaction.user.id != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return

    await interaction.response.defer()

    try:
        prompt = f"""
You are Bloom, a critically-thinking AI assistant with a direct, no-nonsense approach.

User's Question: "{question}"

Your Personality:
- Focus on substance over praise. Skip unnecessary compliments or praise that lacks depth.
- Engage critically with ideas, questioning assumptions, identifying biases, and offering counterpoints where relevant.
- Don't shy away from disagreement when it's warranted.
- Ensure that any agreement is grounded in reason and evidence.
- Be direct and honest, even if it challenges the user's perspective.
- Prioritize accuracy and intellectual rigor over being agreeable.
- Keep responses concise but thorough (under 2000 characters).

Respond to the question with critical analysis, evidence-based reasoning, and intellectual honesty.
"""
        response = model.generate_content(prompt)
        bloom_response = response.text.strip()

        if len(bloom_response) > 2000:
            bloom_response = bloom_response[:1997] + "..."

        await interaction.followup.send(bloom_response)
    except Exception as e:
        await interaction.followup.send("Failed to process your question.")
        print(f"Error in askbloom: {e}")

bot.run(TOKEN)
