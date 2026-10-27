from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """返回 OpenAI 格式的工具定义"""
        return {
            "type": "function",
            "function": {
                "name": "search_course_content",
                "description": "Search course materials with smart course name matching and lesson filtering",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for in the course content"
                        },
                        "course_name": {
                            "type": "string",
                            "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                        },
                        "lesson_number": {
                            "type": "integer",
                            "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        Execute the search tool with given parameters.
        
        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter
            
        Returns:
            Formatted search results or error message
        """
        
        # Use the vector store's unified search interface
        results = self.store.search(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )
        
        # Handle errors
        if results.error:
            return results.error
        
        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."
        
        # Format and return results
        return self._format_results(results)
    
    def _format_results(self, results: SearchResults) -> str:
        """格式化搜索结果，附带课程和课时上下文，并记录来源（含链接）"""
        formatted = []
        sources = []  # 来源列表，每个元素为 {text, url}

        # 缓存课程链接信息，避免重复查询 catalog
        course_links_cache = {}    # course_title -> course_link
        lesson_links_cache = {}    # course_title -> {lesson_number: lesson_link}

        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')

            # 构建上下文标题（给 AI 看）
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # 构建显示文本（给前端展示）
            display_text = course_title
            if lesson_num is not None:
                display_text += f" - Lesson {lesson_num}"

            # 查找链接：优先课时链接，fallback 到课程链接
            url = None
            if lesson_num is not None:
                # 从缓存或查询课时链接
                if course_title not in lesson_links_cache:
                    lesson_links_cache[course_title] = {}
                cached_lesson = lesson_links_cache[course_title].get(lesson_num)
                if cached_lesson is not None:
                    url = cached_lesson
                else:
                    lesson_url = self.store.get_lesson_link(course_title, lesson_num)
                    lesson_links_cache[course_title][lesson_num] = lesson_url
                    url = lesson_url

            # 没有课时链接则尝试课程链接
            if not url:
                if course_title in course_links_cache:
                    url = course_links_cache[course_title]
                else:
                    course_url = self.store.get_course_link(course_title)
                    course_links_cache[course_title] = course_url
                    url = course_url

            sources.append({"text": display_text, "url": url})
            formatted.append(f"{header}\n{doc}")

        # 保存来源供后续获取（去重，同一课时只保留一个）
        seen = set()
        unique_sources = []
        for s in sources:
            key = s["text"]
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)
        self.last_sources = unique_sources

        return "\n\n".join(formatted)

class CourseOutlineTool(Tool):
    """Tool for retrieving the complete outline (lesson list) of a course"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # 追踪来源，供前端展示

    def get_tool_definition(self) -> Dict[str, Any]:
        """返回 OpenAI 格式的工具定义"""
        return {
            "type": "function",
            "function": {
                "name": "get_course_outline",
                "description": "Get the complete outline of a course, including course title, course link, and the full list of lessons with their numbers and titles. Use this when the user asks about course outline, syllabus, curriculum, lesson list, or what topics a course covers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "course_title": {
                            "type": "string",
                            "description": "Course title or partial name (e.g. 'MCP', 'Introduction to AI', 'RAG')"
                        }
                    },
                    "required": ["course_title"]
                }
            }
        }

    def execute(self, course_title: str) -> str:
        """
        获取课程大纲并格式化为字符串返回。

        Args:
            course_title: 课程名称（支持模糊匹配）

        Returns:
            格式化的课程大纲文本
        """
        outline = self.store.get_course_outline(course_title)

        if not outline:
            return f"No course found matching '{course_title}'."

        # 记录来源（用于前端展示）
        self.last_sources = [{
            "text": outline["title"],
            "url": outline.get("course_link")
        }]

        # 格式化输出
        lines = []
        lines.append(f"Course Title: {outline['title']}")
        if outline.get("course_link"):
            lines.append(f"Course Link: {outline['course_link']}")
        if outline.get("instructor"):
            lines.append(f"Instructor: {outline['instructor']}")
        lines.append(f"Total Lessons: {outline.get('lesson_count', 0)}")
        lines.append("")
        lines.append("Lesson List:")
        for lesson in outline.get("lessons", []):
            num = lesson.get("lesson_number", "?")
            title = lesson.get("lesson_title", "Untitled")
            lines.append(f"  Lesson {num}: {title}")

        return "\n".join(lines)


class ToolManager:
    """Manages available tools for the AI"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, tool: Tool):
        """注册实现 Tool 接口的工具"""
        tool_def = tool.get_tool_definition()
        # OpenAI 格式：type: "function", function: { name: ... }
        tool_name = tool_def.get("function", {}).get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"
        
        return self.tools[tool_name].execute(**kwargs)
    
    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources') and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []