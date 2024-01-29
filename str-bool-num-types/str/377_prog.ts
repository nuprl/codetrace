export interface RestApiResponseOutput {
  readonly headers: {
    [key: string]: any;
  };
  readonly body: any;
  readonly statusCode: number;
}

/**
 * API Gateway Response
 */
export class RestApiResponse {

  public cors: boolean;

  constructor() {
    this.cors = true;
  }

  setCors(cors: boolean): void {
    this.cors = !!cors;
  }

  json(body: { [key: string]: any }, statusCode: number = 200) {
    return this.build(JSON.stringify(body), statusCode, 'application/json');
  }

  error(error: any, statusCode: number = 500) {
    let messages;
    if (error instanceof Error) {
      messages = error.toString();
    } else {
      messages = error;
    }
    return this.build(JSON.stringify({
      error: messages,
    }), statusCode ?? error.statusCode ?? 500, 'application/json');
  }

  file(content: any, contentType: string, filename: <FILL>) {
    if (typeof content === 'object') {
      content = JSON.stringify(content, null, 4);
    }
    return this.build(content, 200, contentType, {
      'content-disposition': `attachment; filename=${filename}`,
    });
  }

  notFound() {
    return this.build({}, 404);
  }

  build(
    body: any = {},
    statusCode: number = 200,
    contentType: string = 'application/json',
    headers: { [key: string]: any} = {},
  ): RestApiResponseOutput {
    headers['Content-Type'] = `${contentType};charset=utf-8`;
    if (this.cors) {
      headers['Access-Control-Allow-Origin'] = '*';
      headers['Access-Control-Allow-Headers'] = JSON.stringify([
        'Content-Type',
        'Fetch-Mode',
        'accept',
        'X-Amz-Date',
        'Accept-Encoding',
        'Authorization',
        'X-Api-Key',
        'X-Amz-Security-Token',
        'X-Amz-User-Agent',
      ]);
      headers['Access-Control-Allow-Methods'] = JSON.stringify([
        'HEAD',
        'GET',
        'POST',
        'PUT',
        'DELETE',
      ]);
    }
    return {
      headers,
      body,
      statusCode,
    };
  }
}