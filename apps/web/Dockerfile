FROM node:22-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install

COPY . /app

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
