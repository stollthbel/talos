#!/bin/bash

# Stoll AI Production Deployment Script
# This script handles the complete deployment of the Stoll AI system

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${PROJECT_DIR}/.env"
COMPOSE_FILE="${PROJECT_DIR}/docker-compose.yml"
BACKUP_DIR="${PROJECT_DIR}/backups"

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    log "Docker and Docker Compose are installed"
}

# Check if environment file exists
check_env_file() {
    if [ ! -f "$ENV_FILE" ]; then
        warn "Environment file not found. Creating from template..."
        cp "${PROJECT_DIR}/.env.example" "$ENV_FILE"
        warn "Please edit $ENV_FILE with your configuration before running again."
        exit 1
    fi
    log "Environment file found"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    mkdir -p "${PROJECT_DIR}/data"
    mkdir -p "${PROJECT_DIR}/logs"
    mkdir -p "${PROJECT_DIR}/ssl"
    mkdir -p "$BACKUP_DIR"
    log "Directories created"
}

# Backup existing data
backup_data() {
    if [ -d "${PROJECT_DIR}/data" ] && [ "$(ls -A ${PROJECT_DIR}/data)" ]; then
        log "Backing up existing data..."
        timestamp=$(date +%Y%m%d_%H%M%S)
        backup_path="${BACKUP_DIR}/data_backup_${timestamp}.tar.gz"
        tar -czf "$backup_path" -C "${PROJECT_DIR}" data
        log "Data backed up to $backup_path"
    fi
}

# Build and deploy the application
deploy() {
    log "Starting Stoll AI deployment..."
    
    # Stop existing containers
    log "Stopping existing containers..."
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans
    
    # Build images
    log "Building Docker images..."
    docker-compose -f "$COMPOSE_FILE" build --no-cache
    
    # Start services
    log "Starting services..."
    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Wait for services to be ready
    log "Waiting for services to be ready..."
    sleep 30
    
    # Check health
    check_health
}

# Check service health
check_health() {
    log "Checking service health..."
    
    services=("stoll-ai" "stoll-ai-api" "talos-backend" "redis" "nginx")
    
    for service in "${services[@]}"; do
        if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up"; then
            log "$service is running"
        else
            error "$service is not running"
            docker-compose -f "$COMPOSE_FILE" logs "$service"
        fi
    done
    
    # Test API endpoints
    log "Testing API endpoints..."
    
    # Test Stoll AI API
    if curl -f http://localhost:8001/health &> /dev/null; then
        log "Stoll AI API is healthy"
    else
        warn "Stoll AI API health check failed"
    fi
    
    # Test Talos Backend
    if curl -f http://localhost:5000/health &> /dev/null; then
        log "Talos Backend is healthy"
    else
        warn "Talos Backend health check failed"
    fi
    
    # Test Nginx
    if curl -f http://localhost:80/health &> /dev/null; then
        log "Nginx is healthy"
    else
        warn "Nginx health check failed"
    fi
}

# Show deployment status
show_status() {
    log "Deployment Status:"
    echo
    echo -e "${BLUE}Services:${NC}"
    docker-compose -f "$COMPOSE_FILE" ps
    echo
    echo -e "${BLUE}Service URLs:${NC}"
    echo "  - Stoll AI API: http://localhost:8001"
    echo "  - Talos Backend: http://localhost:5000"
    echo "  - Web Interface: http://localhost:80"
    echo "  - Grafana: http://localhost:3000"
    echo "  - Prometheus: http://localhost:9091"
    echo
    echo -e "${BLUE}Logs:${NC}"
    echo "  - View logs: docker-compose logs -f [service-name]"
    echo "  - View all logs: docker-compose logs -f"
    echo
    echo -e "${BLUE}Management:${NC}"
    echo "  - Stop services: docker-compose down"
    echo "  - Restart service: docker-compose restart [service-name]"
    echo "  - Update: ./deploy.sh --update"
}

# Update deployment
update_deployment() {
    log "Updating Stoll AI deployment..."
    
    # Backup data
    backup_data
    
    # Pull latest images
    log "Pulling latest images..."
    docker-compose -f "$COMPOSE_FILE" pull
    
    # Rebuild and restart
    log "Rebuilding and restarting services..."
    docker-compose -f "$COMPOSE_FILE" up -d --build
    
    # Check health
    check_health
}

# Clean up deployment
cleanup() {
    log "Cleaning up deployment..."
    
    # Stop and remove containers
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes (optional)
    read -p "Remove unused volumes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker volume prune -f
    fi
    
    log "Cleanup completed"
}

# Main execution
main() {
    case "${1:-deploy}" in
        "deploy")
            check_docker
            check_env_file
            create_directories
            backup_data
            deploy
            show_status
            ;;
        "update")
            check_docker
            update_deployment
            show_status
            ;;
        "status")
            show_status
            ;;
        "health")
            check_health
            ;;
        "cleanup")
            cleanup
            ;;
        "logs")
            docker-compose -f "$COMPOSE_FILE" logs -f "${2:-}"
            ;;
        "--help" | "-h")
            echo "Stoll AI Deployment Script"
            echo
            echo "Usage: $0 [command]"
            echo
            echo "Commands:"
            echo "  deploy     - Deploy the application (default)"
            echo "  update     - Update the deployment"
            echo "  status     - Show deployment status"
            echo "  health     - Check service health"
            echo "  cleanup    - Clean up deployment"
            echo "  logs       - Show logs (optionally for specific service)"
            echo "  -h, --help - Show this help message"
            ;;
        *)
            error "Unknown command: $1"
            echo "Use '$0 --help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
