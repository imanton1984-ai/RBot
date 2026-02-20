// webui/src/ws.rs

use axum::{extract::ws::{Message, WebSocket, WebSocketUpgrade}, response::IntoResponse, extract::State};
use futures::{sink::SinkExt, stream::StreamExt};
use serde_json::json;
use crate::state::AppState;

pub async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = state.ws_sender.subscribe();

    let send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            let json_msg = serde_json::to_string(&msg).unwrap_or_else(|_| "{}".into());
            if sender.send(Message::Text(json_msg.into())).await.is_err() { break; }
        }
    });

    let recv_task = tokio::spawn(async move {
        while let Some(Ok(Message::Text(text))) = receiver.next().await {
            if let Ok(msg) = serde_json::from_str::<ClientMessage>(&text) {
                match msg { ClientMessage::Subscribe { channels } => { tracing::info!("Subscribed: {:?}", channels); } }
            }
        }
    });

    tokio::select! { _ = send_task => {}, _ = recv_task => {} }
}

#[derive(Debug, serde::Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    #[serde(rename = "subscribe")]
    Subscribe { channels: Vec<SubscriptionChannel> },
}

#[derive(Debug, serde::Deserialize)]
pub struct SubscriptionChannel { pub name: String, #[serde(default)] pub pair: Option<String>, #[serde(default)] pub tf: Option<i32> }
