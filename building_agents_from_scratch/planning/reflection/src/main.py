from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import openai
import os
import json

@dataclass
class Interaction:
    """Record of a single interaction with the agent"""
    timestamp: datetime
    query: str
    plan: Dict[str, Any]

class Agent:
    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize Agent with empty interaction history."""
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.interactions: List[Interaction] = []  # Working memory
        self.model = model

    def create_system_prompt(self) -> str:
        """Create the system prompt for the LLM with available tools."""
        tools_json = {
            "role": "AI Assistant",
            "capabilities": [
                "Using provided tools to help users when necessary",
                "Responding directly without tools for questions that don't require tool usage",
                "Planning efficient tool usage sequences",
                "If asked by the user, reflecting on the plan and suggesting changes if needed"
            ],
            "instructions": [
                "Use tools only when they are necessary for the task",
                "If a query can be answered directly, respond with a simple message instead of using tools",
                "When tools are needed, plan their usage efficiently to minimize tool calls",
                "If asked by the user, reflect on the plan and suggest changes if needed"
            ],
            "tools": [
                {
                    "name": "convert_currency",
                    "description": "Converts currency using latest exchange rates.",
                    "parameters": {
                        "amount": {
                            "type": "float",
                            "description": "Amount to convert"
                        },
                        "from_currency": {
                            "type": "str",
                            "description": "Source currency code (e.g., USD)"
                        },
                        "to_currency": {
                            "type": "str",
                            "description": "Target currency code (e.g., EUR)"
                        }
                    }
                }
            ],
            "response_format": {
                "type": "json",
                "schema": {
                    "requires_tools": {
                        "type": "boolean",
                        "description": "whether tools are needed for this query"
                    },
                    "direct_response": {
                        "type": "string",
                        "description": "response when no tools are needed",
                        "optional": True
                    },
                    "thought": {
                        "type": "string", 
                        "description": "reasoning about how to solve the task (when tools are needed)",
                        "optional": True
                    },
                    "plan": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "steps to solve the task (when tools are needed)",
                        "optional": True
                    },
                    "tool_calls": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool": {
                                    "type": "string",
                                    "description": "name of the tool"
                                },
                                "args": {
                                    "type": "object",
                                    "description": "parameters for the tool"
                                }
                            }
                        },
                        "description": "tools to call in sequence (when tools are needed)",
                        "optional": True
                    }
                },
                "examples": [
                    {
                        "query": "Convert 100 USD to EUR",
                        "response": {
                            "requires_tools": True,
                            "thought": "I need to use the currency conversion tool to convert USD to EUR",
                            "plan": [
                                "Use convert_currency tool to convert 100 USD to EUR",
                                "Return the conversion result"
                            ],
                            "tool_calls": [
                                {
                                    "tool": "convert_currency",
                                    "args": {
                                        "amount": 100,
                                        "from_currency": "USD", 
                                        "to_currency": "EUR"
                                    }
                                }
                            ]
                        }
                    },
                    {
                        "query": "What's 500 Japanese Yen in British Pounds?",
                        "response": {
                            "requires_tools": True,
                            "thought": "I need to convert JPY to GBP using the currency converter",
                            "plan": [
                                "Use convert_currency tool to convert 500 JPY to GBP",
                                "Return the conversion result"
                            ],
                            "tool_calls": [
                                {
                                    "tool": "convert_currency",
                                    "args": {
                                        "amount": 500,
                                        "from_currency": "JPY",
                                        "to_currency": "GBP"
                                    }
                                }
                            ]
                        }
                    },
                    {
                        "query": "What currency does Japan use?",
                        "response": {
                            "requires_tools": False,
                            "direct_response": "Japan uses the Japanese Yen (JPY) as its official currency. This is common knowledge that doesn't require using the currency conversion tool."
                        }
                    }
                ]
            }
        }
        
        return f"""You are an AI assistant that helps users by providing direct answers or using tools when necessary.
Configuration, instructions, and available tools are provided in JSON format below:

{json.dumps(tools_json, indent=2)}

Always respond with a JSON object following the response_format schema above. 
Remember to use tools only when they are actually needed for the task."""

    def plan(self, user_query: str) -> Dict:
        """Use LLM to create a plan and store it in memory."""
        messages = [
            {"role": "system", "content": self.create_system_prompt()},
            {"role": "user", "content": user_query}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0
        )
        
        try:
            plan = json.loads(response.choices[0].message.content)
            # Store the interaction immediately after planning
            interaction = Interaction(
                timestamp=datetime.now(),
                query=user_query,
                plan=plan
            )
            self.interactions.append(interaction)
            return plan
        except json.JSONDecodeError:
            raise ValueError("Failed to parse LLM response as JSON")

    def reflect_on_plan(self) -> Dict[str, Any]:
        """Reflect on the most recent plan using interaction history."""
        if not self.interactions:
            return {"reflection": "No plan to reflect on", "requires_changes": False}
        
        latest_interaction = self.interactions[-1]
        
        reflection_prompt = {
            "task": "reflection",
            "context": {
                "user_query": latest_interaction.query,
                "generated_plan": latest_interaction.plan
            },
            "instructions": [
                "Review the generated plan for potential improvements",
                "Consider if the chosen tools are appropriate",
                "Verify tool parameters are correct",
                "Check if the plan is efficient",
                "Determine if tools are actually needed"
            ],
            "response_format": {
                "type": "json",
                "schema": {
                    "requires_changes": {
                        "type": "boolean",
                        "description": "whether the plan needs modifications"
                    },
                    "reflection": {
                        "type": "string",
                        "description": "explanation of what changes are needed or why no changes are needed"
                    },
                    "suggestions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "specific suggestions for improvements",
                        "optional": True
                    }
                }
            }
        }
        
        messages = [
            {"role": "system", "content": self.create_system_prompt()},
            {"role": "user", "content": json.dumps(reflection_prompt, indent=2)}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {"reflection": response.choices[0].message.content}

    def execute(self, user_query: str) -> str:
        """Execute the full pipeline: plan, reflect, and potentially replan."""
        try:
            # Create initial plan (this also stores it in memory)
            initial_plan = self.plan(user_query)
            
            # Reflect on the plan using memory
            reflection = self.reflect_on_plan()
            
            # Check if reflection suggests changes
            if reflection.get("requires_changes", False):
                # Generate new plan based on reflection
                messages = [
                    {"role": "system", "content": self.create_system_prompt()},
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": json.dumps(initial_plan)},
                    {"role": "user", "content": f"Please revise the plan based on this feedback: {json.dumps(reflection)}"}
                ]
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0
                )
                
                try:
                    final_plan = json.loads(response.choices[0].message.content)
                except json.JSONDecodeError:
                    final_plan = initial_plan  # Fallback to initial plan if parsing fails
            else:
                final_plan = initial_plan
            
            # Update the stored interaction with all information
            self.interactions[-1].plan = {
                "initial_plan": initial_plan,
                "reflection": reflection,
                "final_plan": final_plan
            }
            
            # Return the appropriate response
            if final_plan.get("requires_tools", True):
                return f"""Initial Thought: {initial_plan['thought']}
Initial Plan: {'. '.join(initial_plan['plan'])}
Reflection: {reflection.get('reflection', 'No improvements suggested')}
Final Plan: {'. '.join(final_plan['plan'])}"""
            else:
                return f"""Response: {final_plan['direct_response']}
Reflection: {reflection.get('reflection', 'No improvements suggested')}"""
            
        except Exception as e:
            return f"Error executing plan: {str(e)}"

def main():
    agent = Agent(model="gpt-4o-mini")
    
    query_list = ["I am traveling to Japan from Lithuania, I have 1500 of local currency, how much of Japaese currency will I be able to get?",
                  "How are you doing?"]
    
    for query in query_list:
        print(f"\nQuery: {query}")
        result = agent.execute(query)
        print(result)

if __name__ == "__main__":
    main()
