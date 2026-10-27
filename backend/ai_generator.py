import json
from openai import OpenAI
from typing import List, Optional, Dict, Any


class AIGenerator:
    """通过 OpenAI 兼容接口与 DeepSeek 交互，生成回答"""

    # 系统提示词
    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
1. **search_course_content**: Search within course content for specific topics, explanations, or details
2. **get_course_outline**: Get the complete outline of a course — title, link, and full lesson list

Tool Usage Guidelines:
- **One tool call per query maximum**
- Use **get_course_outline** when the user asks about a course's outline, syllabus, curriculum, lesson list, topics covered, or course structure
- Use **search_course_content** for questions about specific content, concepts, or details within course material
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Course Outline Response Format:
When answering outline-related queries, always include:
- Course title
- Course link (if available)
- Complete lesson list with each lesson's number and title

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Use the appropriate tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

        # 预构建基础 API 参数
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        生成 AI 回答，支持工具调用和对话上下文。

        Args:
            query: 用户问题
            conversation_history: 之前的对话历史
            tools: AI 可使用的工具列表
            tool_manager: 执行工具的管理器

        Returns:
            生成的回答文本
        """
        # 构建消息列表
        messages = []

        # 系统提示词 + 对话历史
        system_content = self.SYSTEM_PROMPT
        if conversation_history:
            system_content = f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
        messages.append({"role": "system", "content": system_content})

        # 用户消息
        messages.append({"role": "user", "content": query})

        # 准备 API 参数
        api_params: Dict[str, Any] = {
            **self.base_params,
            "messages": messages
        }

        # 添加工具（如果有）
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"

        # 调用 API
        response = self.client.chat.completions.create(**api_params)
        choice = response.choices[0]

        # 处理工具调用
        if choice.message.tool_calls and tool_manager:
            return self._handle_tool_execution(
                choice.message, messages, tool_manager, tools
            )

        # 直接返回回答
        return choice.message.content or ""

    def _handle_tool_execution(self, ai_message, messages: List[Dict[str, Any]],
                               tool_manager, tools) -> str:
        """
        处理工具调用并获取后续回答。

        Args:
            ai_message: 包含工具调用的 AI 消息
            messages: 当前消息列表
            tool_manager: 工具管理器
            tools: 工具定义列表

        Returns:
            工具执行后的最终回答文本
        """
        # 将 AI 的工具调用消息加入对话
        messages.append({
            "role": "assistant",
            "content": ai_message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in ai_message.tool_calls
            ]
        })

        # 执行每个工具调用
        for tool_call in ai_message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            tool_result = tool_manager.execute_tool(func_name, **func_args)

            # 将工具结果加入消息（OpenAI 格式，role 为 tool）
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

        # 第二次调用 API，基于工具结果生成最终回答
        final_params: Dict[str, Any] = {
            **self.base_params,
            "messages": messages
        }

        final_response = self.client.chat.completions.create(**final_params)
        return final_response.choices[0].message.content or ""
