import os
import sqlite3
import json
import math
import requests
from typing import List, Dict, Optional, Tuple
from datetime import timedelta
import discord
from discord import app_commands
from discord.ext import commands
from openai import OpenAI

LEARNING_CHANNEL = '1411335494234669076'
TRADING_CHANNEL = '1418976581099065355'
AUTO_KICK_CHANNEL = '1411335541873709167'
HELPER_ROLES = ['1418434355650625676', '1352853011424219158', '1372300233240739920']
OWNER_ID = '1334138321412296725'

TOKEN = os.getenv('TOKEN')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENAI_API_KEY = os.getenv('AI')

openai_client = OpenAI(api_key=OPENAI_API_KEY)

user_messages = {}

class Database:
    def __init__(self, db_path='bloom.db'):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.create_tables()
    
    def create_tables(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL UNIQUE,
                answer TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def save_knowledge(self, question: str, answer: str, embedding: List[float]) -> bool:
        try:
            self.conn.execute(
                'INSERT OR IGNORE INTO knowledge (question, answer, embedding) VALUES (?, ?, ?)',
                (question, answer, json.dumps(embedding))
            )
            self.conn.commit()
            return self.conn.total_changes > 0
        except Exception as e:
            print(f"Error saving knowledge: {e}")
            return False
    
    def get_all_knowledge(self) -> List[Dict]:
        cursor = self.conn.execute('SELECT * FROM knowledge')
        return [dict(row) for row in cursor.fetchall()]
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0
    
    def find_similar_question(self, question_embedding: List[float], threshold: float = 0.65) -> Optional[Dict]:
        knowledge = self.get_all_knowledge()
        best_match = None
        best_score = 0
        
        for item in knowledge:
            stored_embedding = json.loads(item['embedding'])
            similarity = self.cosine_similarity(question_embedding, stored_embedding)
            
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = {**item, 'similarity': similarity}
        
        return best_match
    
    def save_conversation(self, user_id: str, message: str, response: str):
        self.conn.execute(
            'INSERT INTO conversations (user_id, message, response) VALUES (?, ?, ?)',
            (user_id, message, response)
        )
        self.conn.commit()
    
    def get_conversation_history(self, user_id: str, limit: int = 5) -> List[Dict]:
        cursor = self.conn.execute(
            'SELECT message, response FROM conversations WHERE user_id = ? ORDER BY created_at DESC LIMIT ?',
            (user_id, limit)
        )
        return list(reversed([dict(row) for row in cursor.fetchall()]))

db = Database()

def get_embedding(text: str) -> Optional[List[float]]:
    try:
        response = openai_client.embeddings.create(
            model='text-embedding-3-small',
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def get_openrouter_response(messages: List[Dict], max_tokens: int = 500) -> Optional[str]:
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://replit.com",
                "X-Title": "Bloom Discord Bot"
            },
            json={
                "model": "x-ai/grok-4-fast:free",
                "messages": messages,
                "max_tokens": max_tokens
            }
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            print(f"OpenRouter error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error getting OpenRouter response: {e}")
        return None

def get_ai_response(prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
    messages = [
        {"role": "system", "content": system_prompt or "You are Bloom, a helpful Discord bot assistant."},
        {"role": "user", "content": prompt}
    ]
    return get_openrouter_response(messages)

def get_ai_response_with_history(user_id: str, question: str) -> Optional[str]:
    history = db.get_conversation_history(user_id, 5)
    messages = [
        {
            "role": "system",
            "content": "Focus on substance over praise. Skip unnecessary compliments or praise that lacks depth. Engage critically with ideas, questioning assumptions, identifying biases, and offering counterpoints where relevant. Don't shy away from disagreement when it's warranted, and ensure that any agreement is grounded in reason and evidence."
        }
    ]
    
    for conv in history:
        messages.append({"role": "user", "content": conv['message']})
        messages.append({"role": "assistant", "content": conv['response']})
    
    messages.append({"role": "user", "content": question})
    
    answer = get_openrouter_response(messages)
    if answer:
        db.save_conversation(user_id, question, answer)
    return answer

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.guild_messages = True
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')
    try:
        synced = await bot.tree.sync()
        print(f'Synced {len(synced)} command(s)')
    except Exception as e:
        print(f'Error syncing commands: {e}')

@bot.event
async def on_member_join(member):
    from datetime import datetime, timezone
    
    account_age = datetime.now(timezone.utc) - member.created_at
    
    if account_age.days < 7:
        try:
            await member.send(
                f"Hello {member.name},\n\n"
                f"Your account is less than 7 days old. For security reasons, "
                f"you have been automatically kicked from the server.\n\n"
                f"You are welcome to rejoin once your account is at least 7 days old."
            )
        except:
            pass
        
        try:
            await member.kick(reason="Account < 7 days old")
            
            auto_kick_channel = bot.get_channel(int(AUTO_KICK_CHANNEL))
            if auto_kick_channel:
                kick_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                await auto_kick_channel.send(
                    f"**Moderation Action:** kick\n"
                    f"**Target:** <@{member.id}> (ID: {member.id})\n"
                    f"**By:** Bloom (auto-kick)\n"
                    f"**Reason:** Account < 7 days old\n"
                    f"**Time:** {kick_time}"
                )
        except Exception as e:
            print(f"Error kicking member: {e}")

@bot.event
async def on_message(message):
    if message.author.bot:
        return
    
    if str(message.channel.id) == TRADING_CHANNEL:
        user_id = str(message.author.id)
        content = message.content.strip().lower()
        
        if user_id not in user_messages:
            user_messages[user_id] = []
        
        if content in user_messages[user_id]:
            try:
                await message.delete()
                await message.author.timeout(timedelta(minutes=10), reason="Repeated trade message in trading channel")
                
                temp_msg = await message.channel.send(
                    f"{message.author.mention} You have been muted for 10 minutes for repeating your trade message. "
                    f"Please remember:\n"
                    f"• Do not repeat your trade more than once\n"
                    f"• Maximum 7 lines per trade\n"
                    f"• Take conversations to DMs"
                )
                await temp_msg.delete(delay=10)
                print(f"Muted {message.author.name} for repeating message in trading channel")
            except Exception as e:
                print(f"Error muting user: {e}")
        else:
            user_messages[user_id].append(content)
            if len(user_messages[user_id]) > 10:
                user_messages[user_id].pop(0)
        
        await bot.process_commands(message)
        return
    
    if str(message.channel.id) == LEARNING_CHANNEL:
        if message.reference:
            member = message.author
            has_helper_role = any(str(role.id) in HELPER_ROLES for role in member.roles)
            
            if has_helper_role:
                try:
                    referenced_message = await message.channel.fetch_message(message.reference.message_id)
                    
                    if not referenced_message.author.bot:
                        question = referenced_message.content
                        answer = message.content
                        
                        embedding = get_embedding(question)
                        
                        if embedding:
                            if db.save_knowledge(question, answer, embedding):
                                print(f"Learned new Q&A: \"{question[:50]}...\"")
                except Exception as e:
                    print(f"Error processing helper reply: {e}")
        else:
            question_embedding = get_embedding(message.content)
            
            if question_embedding:
                match = db.find_similar_question(question_embedding, 0.65)
                
                if match:
                    try:
                        await message.reply(match['answer'])
                        print(f"Replied with stored answer ({match['similarity']*100:.1f}% match)")
                    except Exception as e:
                        print(f"Error replying to message: {e}")
    
    await bot.process_commands(message)

@bot.tree.command(name="say", description="Make the bot say something")
@app_commands.describe(message="The message to say")
async def say(interaction: discord.Interaction, message: str):
    await interaction.response.defer(ephemeral=True)
    await interaction.channel.send(message)
    await interaction.followup.send("Message sent!", ephemeral=True)

@bot.tree.command(name="tellmeajoke", description="Get a joke from Bloom")
@app_commands.describe(
    context="Optional context for the joke",
    user="Optional user to mention"
)
async def tellmeajoke(interaction: discord.Interaction, context: Optional[str] = None, user: Optional[discord.User] = None):
    await interaction.response.defer(ephemeral=True)
    
    if context:
        prompt = f"Tell me a unique joke about: {context}. It can be witty, dark, absurd, or edgy. Keep it under 3 sentences. Do not explain the joke."
    else:
        prompt = "Tell me a unique joke. It can be witty, dark, absurd, or edgy. Keep it under 3 sentences. Do not explain the joke."
    
    system_prompt = "You are a humor bot. When this command is used, respond with a joke. The joke can be witty, dark, absurd, or edgy. Keep it under 3 sentences. Do not explain the joke. Each response should be unique and not a repeat of the last one."
    
    joke = get_ai_response(prompt, system_prompt)
    
    if joke:
        final_joke = f"{user.mention} {joke}" if user else joke
        await interaction.channel.send(final_joke)
        await interaction.followup.send("Joke sent!", ephemeral=True)
    else:
        await interaction.followup.send("Failed to generate joke.", ephemeral=True)

@bot.tree.command(name="askbloom", description="Ask Bloom anything")
@app_commands.describe(question="Your question")
async def askbloom(interaction: discord.Interaction, question: str):
    await interaction.response.defer()
    
    answer = get_ai_response_with_history(str(interaction.user.id), question)
    
    if answer:
        await interaction.followup.send(answer)
    else:
        await interaction.followup.send("Sorry, I encountered an error processing your question.")

@bot.tree.command(name="ban", description="Ban a user (Owner only)")
@app_commands.describe(
    user="The user to ban",
    reason="Reason for the ban"
)
async def ban(interaction: discord.Interaction, user: discord.Member, reason: str):
    if str(interaction.user.id) != OWNER_ID:
        await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
        return
    
    await interaction.response.defer(ephemeral=True)
    
    try:
        await user.ban(reason=reason)
        await interaction.followup.send(f"Successfully banned {user.mention} for: {reason}", ephemeral=True)
        
        auto_kick_channel = bot.get_channel(int(AUTO_KICK_CHANNEL))
        if auto_kick_channel:
            from datetime import datetime, timezone
            ban_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            await auto_kick_channel.send(
                f"**Moderation Action:** ban\n"
                f"**Target:** <@{user.id}> (ID: {user.id})\n"
                f"**By:** {interaction.user.mention}\n"
                f"**Reason:** {reason}\n"
                f"**Time:** {ban_time}"
            )
    except Exception as e:
        await interaction.followup.send(f"Failed to ban user: {str(e)}", ephemeral=True)

if __name__ == "__main__":
    bot.run(TOKEN)
