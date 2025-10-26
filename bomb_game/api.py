from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .game import GameManager, create_manager

app = FastAPI(title="Bomb Defusal Coordination Game")
manager: GameManager = create_manager()


class StartResponse(BaseModel):
    game_id: str
    state: dict


class ActionModel(BaseModel):
    agent: str = Field(..., description="Agent name (Alpha, Bravo, Charlie)")
    action: str = Field(..., description="Action type: move, inspect, or cut")
    target: str | None = Field(None, description="Target room for move actions")
    bomb_id: str | None = Field(None, description="Bomb identifier for cut actions")
    color: str | None = Field(None, description="Wire cutter color for cut actions")


class ActionRequest(BaseModel):
    round_actions: list[ActionModel]


class ActionResponse(BaseModel):
    state: dict
    messages: list[str]


@app.post("/game/start", response_model=StartResponse)
def start_game() -> StartResponse:
    state = manager.create_game()
    return StartResponse(game_id=state.game_id, state=manager.serialize_state(state))


@app.get("/game/{game_id}")
def get_state(game_id: str) -> dict:
    try:
        state = manager.get_game(game_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return manager.serialize_state(state)


@app.post("/game/{game_id}/act", response_model=ActionResponse)
def act(game_id: str, request: ActionRequest) -> ActionResponse:
    try:
        state = manager.get_game(game_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        result = manager.apply_actions(state, [action.dict() for action in request.round_actions])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ActionResponse(**result)
