"""
Email service for sending password reset and other notification emails.
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import os
from jinja2 import Template

logger = logging.getLogger(__name__)

class EmailService:
    """Service for sending emails using SMTP configuration from environment."""
    
    def __init__(self):
        """Initialize email service with environment configuration."""
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.from_email = os.getenv("FROM_EMAIL", self.smtp_username)
        self.from_name = os.getenv("FROM_NAME", "Tile Matcher")
        
        if not self.smtp_username or not self.smtp_password:
            logger.warning("Email service not properly configured. SMTP credentials missing.")
    
    def _create_connection(self) -> Optional[smtplib.SMTP]:
        """Create and return authenticated SMTP connection."""
        try:
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls() if self.smtp_use_tls else None
            server.login(self.smtp_username, self.smtp_password)
            return server
        except Exception as e:
            logger.error(f"Failed to create SMTP connection: {e}")
            return None
    
    def send_email(self, to_email: str, subject: str, html_body: str, text_body: Optional[str] = None) -> bool:
        """
        Send an email to the specified recipient.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            html_body: HTML email body
            text_body: Plain text email body (optional)
            
        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        if not self.smtp_username or not self.smtp_password:
            logger.error("Email service not configured. Cannot send email.")
            return False
            
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = to_email
            
            # Add text part if provided
            if text_body:
                text_part = MIMEText(text_body, 'plain')
                msg.attach(text_part)
            
            # Add HTML part
            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)
            
            # Send email
            server = self._create_connection()
            if not server:
                return False
                
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False
    
    def send_password_reset_email(self, to_email: str, reset_token: str, user_name: str) -> bool:
        """
        Send password reset email with reset link.
        
        Args:
            to_email: User's email address
            reset_token: Password reset token
            user_name: User's name
            
        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        # Get frontend URL from environment or use default
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        reset_link = f"{frontend_url}/reset-password?token={reset_token}"
        
        subject = "Reset Your Tile Matcher Password"
        
        # HTML email template
        html_template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Reset Your Password</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .container { max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background-color: #2563eb; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }
                .content { background-color: #f9fafb; padding: 30px; border-radius: 0 0 8px 8px; }
                .button { display: inline-block; background-color: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold; margin: 20px 0; }
                .footer { text-align: center; color: #6b7280; font-size: 14px; margin-top: 30px; }
                .warning { background-color: #fef3c7; border: 1px solid #f59e0b; padding: 15px; border-radius: 6px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔐 Password Reset Request</h1>
                </div>
                <div class="content">
                    <h2>Hello {{ user_name }}!</h2>
                    <p>We received a request to reset your password for your Tile Matcher account.</p>
                    <p>Click the button below to reset your password:</p>
                    
                    <div style="text-align: center;">
                        <a href="{{ reset_link }}" class="button">Reset My Password</a>
                    </div>
                    
                    <p>Or copy and paste this link into your browser:</p>
                    <p style="word-break: break-all; background-color: #e5e7eb; padding: 10px; border-radius: 4px; font-family: monospace;">{{ reset_link }}</p>
                    
                    <div class="warning">
                        <p><strong>Important:</strong> This link will expire in 30 minutes for security reasons.</p>
                    </div>
                    
                    <p>If you didn't request this password reset, you can safely ignore this email. Your password will remain unchanged.</p>
                    
                    <p>For security reasons, this link can only be used once.</p>
                </div>
                <div class="footer">
                    <p>This email was sent by Tile Matcher<br>
                    If you have any questions, please contact our support team.</p>
                </div>
            </div>
        </body>
        </html>
        """)
        
        # Plain text version
        text_template = Template("""
        Hello {{ user_name }}!
        
        We received a request to reset your password for your Tile Matcher account.
        
        Please click on the following link to reset your password:
        {{ reset_link }}
        
        IMPORTANT: This link will expire in 30 minutes for security reasons.
        
        If you didn't request this password reset, you can safely ignore this email. Your password will remain unchanged.
        
        For security reasons, this link can only be used once.
        
        ---
        This email was sent by Tile Matcher
        If you have any questions, please contact our support team.
        """)
        
        html_body = html_template.render(
            user_name=user_name,
            reset_link=reset_link
        )
        
        text_body = text_template.render(
            user_name=user_name,
            reset_link=reset_link
        )
        
        return self.send_email(to_email, subject, html_body, text_body)


# Global instance
email_service = EmailService()