const roomTemplate = `
<!-- <DOCTYPE! html> --->
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Simple Transcriber Client For Mediasoup</title>
</head>
<style>
*
{
  color: #333;
  font-size: 20px;
  font-weight: 300;
  box-sizing: border-box;
  font-family: 'Metrophobic', Arial, serif; font-weight: 400;
}

body {
  background-color: #eeeeee;
  padding: 20px;
}
header,
nav,
section,
aside,
footer
{
  text-align: center;
  background-color: #eeeeee;
  border-radius: 5px;
  padding: 50px;
}
header,
nav,
footer {
  width: 100%;
  padding: 20px;
}
header { margin-bottom: 20px; }
h1 { margin: 0; }
nav,
aside { width: 25%; }
aside ul {
  list-style-type: none;
  padding: 0;
  margin-bottom: 0;
}
aside li {
  display: inline-block;
  margin: 5px 0 5px 0px;
  font-size: 12px;
  width: 100%;
}

section {
  float: left;
  width: calc(50% - 100px);
  margin: 0 50px 50px 50px;
}
section header,
section article,
section footer
{
  width: 100%;
  background-color: #999999;
  border-radius: 5px;
  margin: 0;
  padding: 50px;
}
section article { margin: 25px 0; }
nav { float: left; }
aside { float: right; }
nav ul {
  list-style-type: none;
  padding: 0;
  margin-bottom: 0;
}
nav li {
  display: inline-block;
  margin: 15px 0 15px 0px;
  font-size: 12px;
  width: 100px;
}
nav li:first-child { margin-left: 0; }
footer { clear: both; }
</style>
<body class="hello">
  <nav>
    <h3>Local Client</h3>
    <h4 id="roomId">Room: {roomId}</h4>
    <h4>UserId: <span id="userId"></span></h4>
    <ul id="lablels">
      
    </ul>
    <video id="localVideo" width="160" height="120" autoplay playsinline controls="false"></video>
    <hr />
    <button id="videoController">Pause</button>
    <button id="audioController">Mute</button>
    <script src="./bundled-index.js"></script>
  </nav>
  <section id="remoteClients">
    <h1>Remote Clients</h1>
  </section>
  <aside>
    <h1>Transcriptions</h1>
    <ul id="transcriptions">
      
    </ul>
  </aside>
  <script src="http://{server}:{port}/bundledIndexJs"></script>
</body>
</html>`;

type RoomHtmlOptions = {
  roomId: <FILL>;
  server: string;
  port: number;
};

export function makeRoomHtml(options: RoomHtmlOptions) {
  const result = roomTemplate
    .replace("{roomId}", options.roomId)
    .replace("{server}", options.server)
    .replace("{port}", options.port.toString());
  return result;
}
