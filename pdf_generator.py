import io
import logging
from typing import Dict, Optional
import requests
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import re
from models import Report


def register_cyrillic_fonts():
    try:
        pdfmetrics.registerFont(TTFont('DejaVuSans', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
        pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
        return True
    except:
        try:
            pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
            pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', 'DejaVuSans-Bold.ttf'))
            return True
        except Exception as e:
            logging.warning(f"Не удалось зарегистрировать кириллические шрифты: {e}")
            return False


def create_custom_styles():
    """Создает кастомные стили с поддержкой кириллицы"""
    styles = getSampleStyleSheet()

    font_available = register_cyrillic_fonts()
    font_name = 'DejaVuSans' if font_available else 'Helvetica'
    font_bold = 'DejaVuSans-Bold' if font_available else 'Helvetica-Bold'

    # Заголовок документа
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Title'],
        fontName=font_bold,
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#2c3e50')
    ))

    # Заголовок секции
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading1'],
        fontName=font_bold,
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.HexColor('#34495e')
    ))

    # Заголовок подсекции
    styles.add(ParagraphStyle(
        name='SubSectionHeader',
        parent=styles['Heading2'],
        fontName=font_bold,
        fontSize=14,
        spaceAfter=8,
        spaceBefore=12,
        textColor=colors.HexColor('#7f8c8d')
    ))

    # Обычный текст
    styles.add(ParagraphStyle(
        name='CustomNormal',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10,
        spaceAfter=6,
        alignment=TA_JUSTIFY
    ))

    # Текст инсайтов
    styles.add(ParagraphStyle(
        name='InsightText',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=11,
        spaceAfter=8,
        leftIndent=20,
        alignment=TA_JUSTIFY,
        backColor=colors.HexColor('#f8f9fa'),
        borderColor=colors.HexColor('#dee2e6'),
        borderWidth=1,
        borderPadding=10
    ))

    return styles


def clean_markdown_for_pdf(text: str) -> str:
    """Очищает Markdown разметку для PDF"""
    if not text:
        return ""

    # Убираем **bold** разметку и заменяем на <b>bold</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)

    # Убираем *italic* разметку и заменяем на <i>italic</i>
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)

    # Убираем `code` разметку
    text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', text)

    # Заменяем переносы строк
    text = text.replace('\n', '<br/>')

    return text


def create_correlation_table(correlations: Dict, styles) -> Table:
    """Создает таблицу корреляций"""
    if not correlations:
        return None

    # Подготавливаем данные для таблицы
    table_data = [['Столбец', 'Коррелирует с', 'Коэффициент']]

    for column_name, correlation_data in correlations.items():
        for with_column, coefficient in correlation_data.items():
            if coefficient is not None:
                table_data.append([
                    column_name,
                    with_column,
                    f"{coefficient:.4f}" if isinstance(coefficient, (int, float)) else str(coefficient)
                ])

    if len(table_data) == 1:  # Только заголовок
        return None

    # Создаем таблицу
    table = Table(table_data, colWidths=[2 * inch, 2 * inch, 1.5 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('FONTNAME', (0, 1), (-1, -1), 'DejaVuSans'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    return table


def download_image_from_url(url: str) -> Optional[io.BytesIO]:
    """Скачивает изображение по URL"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return io.BytesIO(response.content)
    except Exception as e:
        logging.error(f"Ошибка при скачивании изображения {url}: {e}")
        return None


def generate_pdf_report(report: Report) -> io.BytesIO:
    """Генерирует PDF отчет"""
    buffer = io.BytesIO()

    try:
        # Создаем документ
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )

        # Получаем стили
        styles = create_custom_styles()
        story = []

        # Заголовок документа
        story.append(Paragraph(f"Отчет по анализу данных #{report.id}", styles['CustomTitle']))
        story.append(Spacer(1, 20))

        # ... остальной код ...

        # Генерируем PDF
        doc.build(story)

        # Проверяем, что buffer содержит данные
        buffer.seek(0)
        content = buffer.read()
        if len(content) == 0:
            raise Exception("PDF документ не был сгенерирован")

        # Возвращаем buffer в начальное положение
        buffer.seek(0)
        return buffer

    except Exception as e:
        logging.error(f"Ошибка при создании PDF: {e}")
        # Создаем минимальный PDF с сообщением об ошибке
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = [
            Paragraph(f"Ошибка при генерации отчета #{report.id}", styles['Title']),
            Paragraph(f"Причина: {str(e)}", styles['Normal'])
        ]
        doc.build(story)
        buffer.seek(0)
        return buffer
