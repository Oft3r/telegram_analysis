from telethon import TelegramClient, types
import google.generativeai as genai
from datetime import datetime, timedelta
import json
from typing import List, Dict
import asyncio
import pandas as pd
import re
from urllib.parse import urlparse

class TelegramAnalyzer:
    def __init__(self, api_id: str, api_hash: str, gemini_api_key: str):
        """
        Inicializa el analizador con las credenciales necesarias
        """
        # Configurar cliente de Telegram
        self.client = TelegramClient('session_name', api_id, api_hash)
        
        # Configurar Gemini
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    async def get_channel_messages(self, channel_url: str, limit: int = 100) -> List[Dict]:
        """
        Obtiene los mensajes más recientes de un canal de Telegram
        """
        try:
            messages = []
            async with self.client:
                channel = await self.client.get_entity(channel_url)
                async for message in self.client.iter_messages(channel, limit=limit):
                    if isinstance(message, types.Message):
                        # Analizar multimedia
                        media_info = await self._analyze_media(message)
                        
                        # Analizar enlaces
                        links = self._extract_links(message.message) if message.message else []
                        
                        # Analizar menciones y hashtags
                        mentions, hashtags = self._extract_mentions_hashtags(message.message) if message.message else ([], [])
                        
                        # Obtener replies de forma segura sin usar get_replies()
                        reply_count = getattr(message.replies, 'replies', 0) if hasattr(message, 'replies') else 0
                        
                        messages.append({
                            'text': message.message or "",
                            'created_at': message.date,
                            'views': message.views if message.views else 0,
                            'forwards': message.forwards if message.forwards else 0,
                            'reply_count': reply_count,
                            'media': media_info,
                            'links': links,
                            'mentions': mentions,
                            'hashtags': hashtags,
                            'raw_message': message
                        })
            return messages
        except Exception as e:
            print(f"Error obteniendo mensajes de {channel_url}: {str(e)}")
            return []

    async def _analyze_media(self, message) -> Dict:
        """
        Analiza el contenido multimedia de un mensaje
        """
        if not message.media:
            return None

        media_info = {
            'type': None,
            'size': None,
            'duration': None,
            'dimensions': None
        }

        try:
            if isinstance(message.media, types.MessageMediaPhoto):
                media_info['type'] = 'photo'
                if hasattr(message, 'photo') and message.photo and hasattr(message.photo, 'sizes'):
                    # Filtrar solo los tamaños que tienen dimensiones
                    valid_sizes = []
                    for size in message.photo.sizes:
                        try:
                            if hasattr(size, 'w') and hasattr(size, 'h'):
                                valid_sizes.append(size)
                            elif hasattr(size, 'width') and hasattr(size, 'height'):
                                # Para PhotoStrippedSize y otros tipos
                                valid_sizes.append(types.PhotoSize(
                                    type=getattr(size, 'type', 'unknown'),
                                    w=getattr(size, 'width', 0),
                                    h=getattr(size, 'height', 0),
                                    size=getattr(size, 'size', 0)
                                ))
                        except Exception:
                            continue

                    if valid_sizes:
                        largest_size = max(valid_sizes, key=lambda x: (getattr(x, 'w', 0) or 0) * (getattr(x, 'h', 0) or 0))
                        media_info['dimensions'] = {
                            'width': getattr(largest_size, 'w', 0) or getattr(largest_size, 'width', 0),
                            'height': getattr(largest_size, 'h', 0) or getattr(largest_size, 'height', 0)
                        }

            elif isinstance(message.media, types.MessageMediaDocument):
                if hasattr(message, 'video') and message.video:
                    media_info['type'] = 'video'
                    media_info['duration'] = getattr(message.video, 'duration', 0)
                    media_info['dimensions'] = {
                        'width': getattr(message.video, 'w', 0),
                        'height': getattr(message.video, 'h', 0)
                    }
                    media_info['size'] = getattr(message.video, 'size', 0)

        except Exception as e:
            print(f"Error al analizar media: {str(e)}")
            media_info['type'] = 'unknown'

        return media_info

    def _extract_links(self, text: str) -> List[str]:
        """
        Extrae enlaces de un texto
        """
        if not text:
            return []
            
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        links = re.findall(url_pattern, text)
        return [{'url': link, 'domain': urlparse(link).netloc} for link in links]

    def _extract_mentions_hashtags(self, text: str) -> tuple:
        """
        Extrae menciones y hashtags de un texto
        """
        if not text:
            return [], []
            
        mentions = re.findall(r'@(\w+)', text)
        hashtags = re.findall(r'#(\w+)', text)
        return mentions, hashtags

    def analyze_messages(self, messages: List[Dict]) -> Dict:
        """
        Analiza un conjunto de mensajes usando Gemini
        """
        if not messages:
            return {"error": "No hay mensajes para analizar"}
        
        # Preparar contexto para Gemini
        messages_text = "\n".join([
            f"Mensaje {i+1} ({message['created_at']}): {message['text']}"
            for i, message in enumerate(messages)
        ])
        
        prompt = f"""
        Analiza los siguientes mensajes de un canal de Telegram y proporciona:
        1. Un resumen general de los temas principales discutidos
        2. El tono y estilo de comunicación usado en el canal
        3. Patrones en la forma de compartir información
        4. Posibles objetivos o propósito del canal basado en el contenido
        5. Temas recurrentes o palabras clave frecuentes
        6. Evalúa específicamente:
           - La actividad y engagement del canal
           - La calidad y utilidad del contenido
           - La generosidad (sorteos, regalos, consejos gratuitos)
           - La confiabilidad y credibilidad del canal
        
        Mensajes a analizar:
        {messages_text}
        """
        
        try:
            response = self.model.generate_content(prompt)
            return {
                "analysis": response.text,
                "message_count": len(messages),
                "date_range": {
                    "start": min(m['created_at'] for m in messages),
                    "end": max(m['created_at'] for m in messages)
                }
            }
        except Exception as e:
            return {"error": f"Error en el análisis: {str(e)}"}

    async def analyze_multiple_channels(self, channel_urls: List[str], messages_per_channel: int = 100) -> Dict:
        """
        Analiza mensajes de múltiples canales
        """
        results = {}
        
        async with self.client:
            for channel_url in channel_urls:
                print(f"Analizando mensajes de {channel_url}...")
                messages = await self.get_channel_messages(channel_url, messages_per_channel)
                analysis = self.analyze_messages(messages)
                
                results[channel_url] = {
                    "messages": messages,
                    "analysis": analysis
                }
        
        return results

    def generate_report(self, analysis_results: Dict) -> str:
        """
        Genera un reporte en formato markdown con los resultados
        """
        report = "# Análisis de Canales de Telegram\n\n"
        
        for channel_url, data in analysis_results.items():
            report += f"## Canal: {channel_url}\n\n"
            
            if "error" in data["analysis"]:
                report += f"Error: {data['analysis']['error']}\n\n"
                continue
            
            report += "### Resumen del Análisis\n"
            report += f"{data['analysis']['analysis']}\n\n"
            
            report += "### Estadísticas\n"
            report += f"- Mensajes analizados: {data['analysis']['message_count']}\n"
            report += f"- Período: {data['analysis']['date_range']['start']} a {data['analysis']['date_range']['end']}\n\n"
            
            report += "### Mensajes Más Destacados\n"
            # Ordenar por engagement (views + forwards)
            top_messages = sorted(
                data['messages'],
                key=lambda x: x['views'] + x['forwards'],
                reverse=True
            )[:5]
            
            for msg in top_messages:
                report += f"- {msg['text'][:200]}...\n"
                report += f"  - Vistas: {msg['views']}, Reenvíos: {msg['forwards']}\n"
                report += f"  - Respuestas: {msg['reply_count']}\n"
                report += f"  - Fecha: {msg['created_at']}\n"
                channel_name = channel_url.split('/')[-1]
                msg_id = getattr(msg.get('raw_message', {}), 'id', '')
                if msg_id:
                    report += f"  - Ver mensaje original: {channel_url}/{msg_id}\n"
                report += "\n"
            
            report += "### Enlaces Más Compartidos\n"
            for domain, count in data['metrics']['top_domains'].items():
                report += f"- {domain}: {count} veces\n"
            report += "\n"
            
            report += "### Hashtags Más Usados\n"
            for hashtag, count in data['metrics']['top_hashtags'].items():
                report += f"- #{hashtag}: {count} veces\n"
            report += "\n"
            
            report += "### Menciones Más Frecuentes\n"
            for mention, count in data['metrics']['top_mentions'].items():
                report += f"- @{mention}: {count} veces\n"
            report += "\n"
            
            report += "### Calificaciones del Canal\n"
            ratings = data['metrics']['ratings']
            explanations = ratings['explanations']
            
            for metric, label in [
                ('activity', 'Actividad del Canal'),
                ('content_quality', 'Calidad del Contenido'),
                ('generosity', 'Generosidad'),
                ('trustworthy', 'Confiabilidad')
            ]:
                score = float(ratings[metric])  # Asegurar que es número
                explanation = explanations[metric]
                warning = f" ⚠️ *{explanation}*" if score < 5.0 and explanation else ""
                report += f"- {label}: {score}/10{warning}\n"
            
            report += "\n"
        
        return report

    def calculate_channel_metrics(self, messages: List[Dict]) -> Dict:
        """
        Calcula métricas adicionales sobre el canal
        """
        if not messages:
            return {}
        
        df = pd.DataFrame(messages)
        
        # Análisis de multimedia
        media_counts = {
            'photos': sum(1 for m in messages if m['media'] and m['media']['type'] == 'photo'),
            'videos': sum(1 for m in messages if m['media'] and m['media']['type'] == 'video')
        }
        
        # Análisis de enlaces
        all_domains = [link['domain'] for msg in messages for link in msg['links']]
        top_domains = pd.Series(all_domains).value_counts().head(5).to_dict() if all_domains else {}
        
        # Análisis de hashtags y menciones
        all_hashtags = [tag for msg in messages for tag in msg['hashtags']]
        all_mentions = [mention for msg in messages for mention in msg['mentions']]
        
        return {
            "total_views": df['views'].sum(),
            "average_views": df['views'].mean(),
            "total_forwards": df['forwards'].sum(),
            "total_replies": df['reply_count'].sum(),
            "messages_per_day": len(messages) / (max(df['created_at']) - min(df['created_at'])).days,
            "peak_hours": df.groupby(df['created_at'].dt.hour)['views'].mean().sort_values(ascending=False).head(3).to_dict(),
            'media_stats': media_counts,
            'top_domains': top_domains,
            'top_hashtags': pd.Series(all_hashtags).value_counts().head(5).to_dict(),
            'top_mentions': pd.Series(all_mentions).value_counts().head(5).to_dict()
        }

    def calculate_channel_ratings(self, messages: List[Dict], analysis_text: str) -> Dict:
        """
        Calcula las calificaciones del canal en diferentes aspectos
        """
        if not messages:
            return {}
        
        df = pd.DataFrame(messages)
        
        # Calcular actividad
        msgs_per_day = len(messages) / max((max(df['created_at']) - min(df['created_at'])).days, 1)
        avg_engagement = df['views'].mean() + df['forwards'].mean()
        activity_score = min(10, (msgs_per_day * 2 + avg_engagement / 1000) / 3)
        
        # Análisis de contenido para calidad y confiabilidad
        positive_keywords = ['útil', 'gracias', 'excelente', 'bueno', 'recomendado', 'verified', 'oficial']
        reward_keywords = ['sorteo', 'regalo', 'gratis', 'recompensa', 'airdrop', 'premio']
        
        # Análisis de texto para las calificaciones
        text_lower = analysis_text.lower()
        
        # Contar menciones de palabras clave
        quality_mentions = sum(1 for word in positive_keywords if word in text_lower)
        reward_mentions = sum(1 for word in reward_keywords if word in text_lower)
        
        # Calcular puntuaciones
        content_quality = min(10, quality_mentions * 2 + avg_engagement / 2000)
        generosity = min(10, reward_mentions * 2.5)
        trustworthy = min(10, quality_mentions * 1.5 + (df['forwards'].mean() / 100))
        
        # Generar explicaciones para calificaciones bajas
        explanations = {
            "activity": "Poca frecuencia de publicación y bajo engagement" if activity_score < 5 else "",
            "content_quality": "Contenido poco relevante o de bajo valor" if content_quality < 5 else "",
            "generosity": "Pocas recompensas o contenido gratuito" if generosity < 5 else "",
            "trustworthy": "Bajo nivel de credibilidad y confianza" if trustworthy < 5 else ""
        }
        
        return {
            "activity": round(activity_score, 1),
            "content_quality": round(content_quality, 1),
            "generosity": round(generosity, 1),
            "trustworthy": round(trustworthy, 1),
            "explanations": explanations
        }

def print_ascii_header():
    """
    Imprime un encabezado decorativo en ASCII
    """
    print("""
╔════════════════════════════════════════════╗
║        TELEGRAM CHANNEL ANALYZER           ║
║        By Your Developer Oft3r             ║
╚════════════════════════════════════════════╝
    """)

def print_section_header(text):
    """
    Imprime un encabezado de sección decorativo
    """
    print(f"\n{'='*50}")
    print(f"║ {text}")
    print(f"{'='*50}\n")

def print_analysis_results(channel_url, data):
    """
    Imprime los resultados del análisis en consola
    """
    print_section_header(f"Análisis del Canal: {channel_url}")
    
    # Métricas básicas
    metrics = data['metrics']
    print("📊 MÉTRICAS GENERALES:")
    print(f"├─ Total de vistas: {metrics['total_views']:,}")
    print(f"├─ Promedio de vistas: {metrics['average_views']:.2f}")
    print(f"├─ Total de reenvíos: {metrics['total_forwards']:,}")
    print(f"└─ Mensajes por día: {metrics['messages_per_day']:.2f}")

    # Enlaces más compartidos
    print("\n🔗 TOP DOMINIOS COMPARTIDOS:")
    for domain, count in metrics['top_domains'].items():
        print(f"├─ {domain}: {count}")

    # Hashtags
    print("\n#️⃣ HASHTAGS MÁS USADOS:")
    for hashtag, count in metrics['top_hashtags'].items():
        print(f"├─ #{hashtag}: {count}")

    # Menciones
    print("\n👤 MENCIONES MÁS FRECUENTES:")
    for mention, count in metrics['top_mentions'].items():
        print(f"├─ @{mention}: {count}")

    # Agregar sección de calificaciones
    print("\n⭐ CALIFICACIONES DEL CANAL:")
    ratings = data['metrics']['ratings']
    explanations = ratings['explanations']
    
    for metric, label in [
        ('activity', 'Actividad del Canal'),
        ('content_quality', 'Calidad del Contenido'),
        ('generosity', 'Generosidad'),
        ('trustworthy', 'Confiabilidad')
    ]:
        score = float(ratings[metric])  # Asegurar que es número
        explanation = explanations[metric]
        prefix = "└─" if metric == "trustworthy" else "├─"
        warning = f" ⚠️  {explanation}" if score < 5.0 and explanation else ""
        print(f"{prefix} {label}: {score}/10{warning}")

    # Análisis de contenido
    print("\n📝 ANÁLISIS DE CONTENIDO:")
    analysis_text = data['analysis']['analysis'].replace('\n', '\n│ ')
    print(f"│ {analysis_text}")

async def main():
    # Configurar credenciales
    API_ID = "tu_api_id"
    API_HASH = "tu_api_hash"
    GEMINI_API_KEY = "tu_api_key_de_gemini"
    
    print_ascii_header()
    
    # Solicitar URL del canal
    channel_url = input("\nIngrese la URL del canal de Telegram a analizar: ").strip()
    
    # Crear instancia del analizador
    analyzer = TelegramAnalyzer(API_ID, API_HASH, GEMINI_API_KEY)
    
    print("\n🔄 Analizando canal... Por favor espere...")
    
    try:
        # Realizar análisis
        results = await analyzer.analyze_multiple_channels([channel_url])
        
        if not results or channel_url not in results:
            print("❌ Error: No se pudieron obtener los mensajes del canal")
            return
        
        channel_data = results[channel_url]
        if not channel_data.get('messages'):
            print("❌ Error: No hay mensajes para analizar")
            return
        
        # Primero realizar el análisis
        analysis = analyzer.analyze_messages(channel_data['messages'])
        if isinstance(analysis, dict) and "error" in analysis:
            print(f"❌ Error en el análisis: {analysis['error']}")
            return
        
        # Luego calcular métricas
        metrics = analyzer.calculate_channel_metrics(channel_data['messages'])
        
        # Asegurarse de que analysis['analysis'] existe
        analysis_text = analysis.get('analysis', '')
        if not analysis_text:
            print("❌ Error: El análisis no generó resultados")
            return
            
        # Calcular ratings
        metrics['ratings'] = analyzer.calculate_channel_ratings(
            channel_data['messages'],
            analysis_text
        )
        
        # Actualizar los resultados
        results[channel_url].update({
            'metrics': metrics,
            'analysis': analysis
        })
        
        # Mostrar resultados
        print_analysis_results(channel_url, results[channel_url])
        
        # Generar y guardar reporte
        report = analyzer.generate_report(results)
        with open("telegram_analysis_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        print("\n✅ Análisis completado!")
        print("📄 El reporte detallado se ha guardado en 'telegram_analysis_report.md'")
        
    except Exception as e:
        print(f"❌ Error durante el análisis: {str(e)}")
        print("Detalles del error:", e.__class__.__name__)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
