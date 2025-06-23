#!/bin/bash

# Oracle Cloud Deployment Script for InstaMood API

echo "🚀 Starting Oracle Cloud deployment for InstaMood..."

# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker if not already installed
if ! command -v docker &> /dev/null; then
    echo "📦 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

# Install Docker Compose if not already installed
if ! command -v docker-compose &> /dev/null; then
    echo "📦 Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Stop and remove existing containers
echo "🛑 Stopping existing containers..."
sudo docker-compose down 2>/dev/null || true

# Build and start the application
echo "🔨 Building and starting InstaMood API..."
sudo docker-compose up --build -d

# Show status
echo "📊 Deployment status:"
sudo docker-compose ps

echo "✅ InstaMood API deployed successfully!"
echo "🌐 Your API should be available at: http://$(curl -s ifconfig.me):8000"
echo "📖 API Documentation: http://$(curl -s ifconfig.me):8000/docs"
echo ""
echo "🔍 To check logs: sudo docker-compose logs -f"
echo "🔄 To restart: sudo docker-compose restart"
echo "🛑 To stop: sudo docker-compose down" 