# ---- Build stage ----
FROM node:20-alpine AS build
 
WORKDIR /app
 
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL
 
COPY frontend/package.json ./
RUN npm install
 
COPY frontend/ .
RUN npm run build
 
# ---- Serve stage ----
FROM nginx:alpine
 
COPY --from=build /app/dist /usr/share/nginx/html
COPY deployment/nginx/nginx.conf /etc/nginx/conf.d/default.conf
 
EXPOSE 80

HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=10s \
  CMD wget -O /dev/null -q http://localhost:80 || exit 1
 
CMD ["nginx", "-g", "daemon off;"]
