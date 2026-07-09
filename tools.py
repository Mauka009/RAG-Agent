import datetime
import requests
import random
import re
import jieba
from langchain.tools import tool

# 导入你的 RAG 检索函数（请确认路径和函数名正确）
from day7_day3plusMax import hybrid_search_with_rerank

@tool
def rag_search(query: str) -> str:
    """当你需要从内部知识库或文档中查找信息时，使用此工具。适用于政策查询、业务规则、技术文档等场景。"""
    # 延迟导入：只有真正调用时才导入 RAG 模块
    try:
        from day7_day3plusMax import hybrid_search_with_rerank
    except ImportError as e:
        return f"RAG 模块导入失败: {e}，请检查环境依赖。"
    
    try:
        docs = hybrid_search_with_rerank(query, top_k=3, use_bm25=True, use_rerank=True)
        if not docs:
            return "未找到相关信息，请尝试换个问题。"
        result_parts = []
        for doc in docs[:3]:
            content = doc.page_content
            if len(content) > 500:
                content = content[:500] + "..."
            result_parts.append(f"- {content}")
        return "\n".join(result_parts)
    except Exception as e:
        return f"RAG检索失败：{str(e)}"

@tool
def calculator(expression: str) -> str:
    """当你需要进行数学计算时，使用此工具。输入一个数学表达式，返回计算结果。"""
    try:
        # 只允许数字、运算符、括号和空格
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "错误：表达式包含非法字符。"
        # 安全计算
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

@tool
def get_current_time(query: str = "") -> str:
    """当你需要知道当前时间时，使用此工具。返回当前的日期和时间。"""
    now = datetime.datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")

import requests
from langchain.tools import tool

# ---------- 天气工具（真实API） ----------
@tool
def get_weather(city: str = "北京") -> str:
    """当你需要查询某个城市的天气时，使用此工具。输入城市名称，返回实时天气信息。"""
    try:
        # 1. 地理编码：把城市名转成经纬度
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {"name": city, "count": 1, "language": "zh", "format": "json"}
        geo_resp = requests.get(geo_url, params=geo_params, timeout=10)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        
        if not geo_data.get("results"):
            return f"未找到城市：{city}，请检查输入。"
        
        location = geo_data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        location_name = location.get("name", city)
        country = location.get("country", "")
        
        # 2. 获取天气
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "timezone": "auto",
            "forecast_days": 1
        }
        weather_resp = requests.get(weather_url, params=weather_params, timeout=10)
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()
        
        current = weather_data.get("current", {})
        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")
        weather_code = current.get("weather_code")
        
        # 3. 天气代码转文字
        weather_map = {
            0: "晴天", 1: "主要晴天", 2: "局部多云", 3: "阴天",
            45: "雾", 48: "霜雾",
            51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
            61: "小雨", 63: "中雨", 65: "大雨",
            71: "小雪", 73: "中雪", 75: "大雪",
            80: "阵雨", 81: "中阵雨", 82: "大阵雨",
            95: "雷暴", 96: "雷暴加小冰雹", 99: "雷暴加大冰雹"
        }
        weather_desc = weather_map.get(weather_code, f"未知天气代码({weather_code})")
        
        # 4. 组装返回信息
        result = f"📍 {location_name}"
        if country:
            result += f", {country}"
        result += f"\n🌡️ 温度：{temp}°C"
        result += f"\n💧 湿度：{humidity}%"
        result += f"\n🌬️ 风速：{wind} km/h"
        result += f"\n☁️ 天气：{weather_desc}"
        
        return result
        
    except requests.exceptions.Timeout:
        return "天气服务请求超时，请稍后重试。"
    except requests.exceptions.RequestException as e:
        return f"天气服务请求失败：{str(e)}"
    except Exception as e:
        return f"获取天气信息失败：{str(e)}"