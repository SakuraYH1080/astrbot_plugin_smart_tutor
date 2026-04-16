"""智能助教系统插件。"""

# pyright: reportMissingImports=false

import asyncio
import json
import time

import aiosqlite

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Plain
from astrbot.api.star import Context, Star, StarTools, register


@register(
    name="智能助教系统",
    author="待填写",
    desc="课后习题批改与答疑插件模板",
    version="0.1.0",
    repo="",
)
class SmartTutorPlugin(Star):
    """智能助教系统插件主类。

    这里实现一个基础的多模态助教入口：
    1. 提取用户输入中的纯文本和图片。
    2. 空消息直接提示用户补充内容。
    3. 调用当前会话使用的 LLM。
    4. 将模型回答以纯文本形式返回。
    """

    SYSTEM_PROMPT = (
        "你是一名严谨、耐心的中文智能助教。"
        "你需要帮助学生批改作业、讲解题目、分析思路，并在必要时给出分步骤提示。"
        "请优先使用中文回答，表达清晰、结构化，避免空泛套话。"
        "如果题目存在歧义，请先指出歧义并给出最可能的解释。"
        "如果用户提供了图片，请结合图片内容一起分析。"
    )

    def __init__(self, context: Context):
        super().__init__(context)
        self.context = context
        self.db: aiosqlite.Connection | None = None
        self.init_lock = asyncio.Lock()
        self.write_lock = asyncio.Lock()
        self.db_path = StarTools.get_data_dir() / "tutor_records.db"
        logger.info("智能助教系统插件已初始化")

    async def init_db(self) -> None:
        """初始化本地 SQLite 数据库与表结构。"""
        if self.db is not None:
            return

        async with self.init_lock:
            if self.db is not None:
                return

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.db = await aiosqlite.connect(self.db_path)
            await self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS tutor_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    question_content TEXT NOT NULL,
                    bot_reply TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await self.db.commit()
            logger.info("智能助教数据库已初始化: %s", self.db_path)

    async def save_record(
        self, user_id: str, question_content: str, bot_reply: str
    ) -> None:
        """保存一次助教问答记录。"""
        await self.init_db()

        if self.db is None:
            raise RuntimeError("数据库连接未初始化")

        async with self.write_lock:
            await self.db.execute(
                """
                INSERT INTO tutor_records (user_id, question_content, bot_reply)
                VALUES (?, ?, ?)
                """,
                (user_id, question_content, bot_reply),
            )
            await self.db.commit()
            logger.info(
                "智能助教记录已保存: user_id=%s saved_at=%s",
                user_id,
                time.time(),
            )

    async def terminate(self):
        """插件卸载时释放全局数据库连接资源。"""
        if self.db is not None:
            await self.db.close()
            self.db = None
            logger.info("数据库连接已安全关闭")

    @staticmethod
    def _extract_text_and_images(event: AstrMessageEvent) -> tuple[str, list[str]]:
        """从消息链中提取纯文本和图片输入。"""
        text_parts: list[str] = []
        image_inputs: list[str] = []

        for component in event.get_messages():
            if isinstance(component, Plain):
                text = str(getattr(component, "text", "") or "").strip()
                if text:
                    text_parts.append(text)
            elif isinstance(component, Image):
                image_url = getattr(component, "url", None) or getattr(
                    component, "file", None
                )
                if not image_url:
                    continue
                image_inputs.append(str(image_url))

        return "\n".join(text_parts).strip(), image_inputs

    @staticmethod
    def _build_question_content(user_text: str, image_inputs: list[str]) -> str:
        """将问题文本和图片链接序列化，便于后续迁移到关系型数据库。"""
        payload = {"text": user_text, "image_urls": image_inputs}
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _resolve_user_id(event: AstrMessageEvent, group_id: str) -> str:
        """优先保存 QQ 号，其次保存群号。"""
        sender = getattr(event.message_obj, "sender", None)
        sender_user_id = str(getattr(sender, "user_id", "") or "").strip()
        if sender_user_id:
            return sender_user_id

        if group_id:
            return group_id

        return event.unified_msg_origin

    @filter.command("tutor", alias={"助教"})
    async def tutor(self, event: AstrMessageEvent):
        """处理 `/tutor` 和 `/助教` 命令。"""
        await self.init_db()

        user_text, image_inputs = self._extract_text_and_images(event)

        if not user_text and not image_inputs:
            yield event.plain_result(
                "请在 /tutor 或 /助教 后面发送题目文字，或者直接附上题目图片。"
            )
            return

        try:
            provider_id = await self.context.get_current_chat_provider_id(
                umo=event.unified_msg_origin
            )

            llm_resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=user_text or None,
                image_urls=image_inputs or None,
                system_prompt=self.SYSTEM_PROMPT,
            )

            answer_text = (llm_resp.completion_text or "").strip()
            if not answer_text:
                answer_text = "抱歉，这次没有生成可用答案，请稍后再试一次。"
            else:
                group_id = event.get_group_id()
                user_id = self._resolve_user_id(event, group_id)
                question_content = self._build_question_content(user_text, image_inputs)
                await self.save_record(user_id, question_content, answer_text)

            yield event.plain_result(answer_text)
        except Exception as exc:
            logger.exception("智能助教调用 LLM 失败: %s", exc)
            yield event.plain_result("抱歉，助教的大脑暂时卡壳了，请稍后再试一下哦。")
